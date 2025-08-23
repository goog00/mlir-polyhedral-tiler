//===- GpuTileILPSelectionPass.cpp - 通过多面体信息驱动的分块选择 -*- C++
//-*-===//
//
// 目标:
//   将多面体分析得到的迭代域与依赖提示融合进整数线性规划模型,
//   使用 OR-Tools 的 CP-SAT 求解分块大小集合 {T_i}, 并写回属性 "tile.sizes".
//
// 多面体信息的获取途径(两级回退):
//   1) 首选: 读取其它分析 Pass 写入的属性, 例如:
//      - "poly.extents"            : ArrayAttr(IntegerAttr)
//      表示每层理论迭代次数上界 E_i.
//      - "poly.depend.dirs"        : ArrayAttr(StringAttr)
//      表示每层是否存在严格方向("<",">"),
//                                    或 "=","*" 等提示, 用于调权或收紧上界.
//      - "poly.sm.coeffs"          : ArrayAttr(IntegerAttr)
//      表示共享内存线性系数 c_i. 若存在则直接采用, 使该 Pass 与你的
//      BuildPolyRepresent/BuildPolyDepend 对接.
//   2) 回退: 若未提供属性, 则在本 Pass 内做保守估计:
//      - 从 affine.for 的上下界常量解析 E_i, 失败则使用回退上界.
//      - 遍历带内的 AffineLoad/AffineStore, 对每个 memref
//      用线性化的保守系数估计共享内存.
//      - 依赖提示采用“越内层权重越大”的稳健缺省.
//
 
//
//===----------------------------------------------------------------------===//

#include <optional>
#include <string>
#include <vector>

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/Plugins/PassPlugin.h"
#include "ortools/sat/cp_model.h"

using namespace mlir;

namespace {

static void collectPerfectNestedBand(affine::AffineForOp outerMost,
                                     SmallVector<affine::AffineForOp, 8> &band) {
    band.clear();
    affine::AffineForOp current = outerMost;
    while (true) {
        band.push_back(current);
        Block &block = current.getRegion().front();

        affine::AffineForOp uniqueChild = nullptr;
        for (Operation &operation : block) {
            if (auto forOperation = dyn_cast<affine::AffineForOp>(&operation)) {
                if (uniqueChild != nullptr) {
                    uniqueChild = nullptr;
                    break;
                } else {
                    uniqueChild = forOperation;
                }
            }
        }

        bool onlyLoopAndYield = false;
        if (uniqueChild != nullptr) {
            Operation &firstOperation = block.front();
            Operation &lastOperation = block.back();
            if (&firstOperation == uniqueChild.getOperation() &&
                isa<affine::AffineYieldOp>(&lastOperation)) {
                onlyLoopAndYield = true;
            }
        }

        if (uniqueChild != nullptr && onlyLoopAndYield) {
            current = uniqueChild;
            continue;
        } else {
            break;
        }
    }
}

static std::optional<int64_t> tryComputeConstantExtent(affine::AffineForOp loop) {
    int64_t step = 1;
    {
        auto stepAPInt = loop.getStep();
        step = stepAPInt.getSExtValue();
        if (step <= 0) {
            step = 1;
        }
    }

    AffineMap lowerBoundMap = loop.getLowerBoundMap();
    AffineMap upperBoundMap = loop.getUpperBoundMap();

    if (lowerBoundMap.getNumDims() == 0 && lowerBoundMap.getNumSymbols() == 0 &&
        upperBoundMap.getNumDims() == 0 && upperBoundMap.getNumSymbols() == 0) {
        if (lowerBoundMap.getNumResults() == 1 && upperBoundMap.getNumResults() == 1) {
            auto lowerExpr = llvm::dyn_cast<AffineConstantExpr>(lowerBoundMap.getResult(0));
            auto upperExpr = llvm::dyn_cast<AffineConstantExpr>(upperBoundMap.getResult(0));
            if (lowerExpr && upperExpr) {
                int64_t lower = lowerExpr.getValue();
                int64_t upper = upperExpr.getValue();
                if (upper <= lower) {
                    return 0;
                }
                int64_t distance = upper - lower;
                int64_t extent = (distance + step - 1) / step;
                return extent;
            }
        }
    }
    return std::nullopt;
}

/// 从属性 "poly.extents" 中读取每层上界, 若不存在则回退到常量解析.
static std::vector<int64_t> getLayerExtentsFromAttributesOrFallback(
    const SmallVector<affine::AffineForOp, 8> &band, int64_t fallbackExtentUpperBound) {
    std::vector<int64_t> extents;
    extents.resize(band.size(), fallbackExtentUpperBound);

    affine::AffineForOp outerMost = band.front();
    if (auto array = outerMost->getAttrOfType<ArrayAttr>("poly.extents")) {
        if (array.size() == band.size()) {
            for (size_t i = 0; i < band.size(); ++i) {
                if (auto intAttr = array[i].dyn_cast<IntegerAttr>()) {
                    int64_t v = intAttr.getInt();
                    if (v > 0) {
                        extents[i] = v;
                    }
                }
            }
            return extents;
        }
    }

    for (size_t i = 0; i < band.size(); ++i) {
        auto maybeExtent = tryComputeConstantExtent(band[i]);
        if (maybeExtent.has_value() && maybeExtent.value() > 0) {
            extents[i] = maybeExtent.value();
        }
    }
    return extents;
}

/// 从属性 "poly.depend.dirs" 中提取每层方向提示, 形成每层的收紧上界与权重修饰.
/// 约定: 该属性为与层数相同的 ArrayAttr(StringAttr), 每个字符串包含符号集合,
/// 例如 "=,*" 或 "<,=,>".
static void getDependHintsFromAttributesOrFallback(const SmallVector<affine::AffineForOp, 8> &band,
                                                   /*输出*/ std::vector<int64_t> &layerUpperCap,
                                                   /*输出*/ std::vector<int64_t> &layerWeights,
                                                   int64_t defaultUpperCap,
                                                   int64_t innermostBonus) {
    size_t depth = band.size();
    layerUpperCap.assign(depth, defaultUpperCap);
    layerWeights.assign(depth, 1);

    for (size_t i = 0; i < depth; ++i) {
        layerWeights[i] = static_cast<int64_t>(i + 1);
    }
    if (depth > 0) {
        layerWeights[depth - 1] += innermostBonus;
    }

    affine::AffineForOp outerMost = band.front();
    if (auto array = outerMost->getAttrOfType<ArrayAttr>("poly.depend.dirs")) {
        if (array.size() == depth) {
            for (size_t i = 0; i < depth; ++i) {
                StringRef dir = "";
                if (auto s = array[i].dyn_cast<StringAttr>()) {
                    dir = s.getValue();
                }
                bool hasStrict = dir.contains('<') || dir.contains('>');
                bool hasEqual = dir.contains('=');
                if (hasStrict) {
                    // 若存在严格方向, 收紧该层上界, 并减小权重.
                    layerUpperCap[i] = std::min<int64_t>(layerUpperCap[i], 64);
                    if (layerWeights[i] > 1) {
                        layerWeights[i] -= 1;
                    }
                }
                if (hasEqual) {
                    // 方向为等号说明有复用, 提高权重.
                    layerWeights[i] += 1;
                }
            }
        }
    }
}

/// 从属性 "poly.sm.coeffs" 中读取共享内存线性系数, 若不存在则回退为保守估计.
static std::vector<int64_t> getSharedMemoryCoefficientsOrFallback(
    const SmallVector<affine::AffineForOp, 8> &band,
    /*输出*/ int64_t &baseBytes) {
    affine::AffineForOp outerMost = band.front();
    if (auto array = outerMost->getAttrOfType<ArrayAttr>("poly.sm.coeffs")) {
        std::vector<int64_t> coeffs(array.size(), 0);
        for (size_t i = 0; i < array.size(); ++i) {
            if (auto ia = array[i].dyn_cast<IntegerAttr>()) {
                coeffs[i] = ia.getInt();
            }
        }
        baseBytes = 0;
        return coeffs;
    }

    // 回退: 在带内遍历 load/store, 对每个 memref 给出保守线性化系数.
    llvm::DenseSet<Value> visitedBuffers;
    baseBytes = 0;
    size_t depth = band.size();
    std::vector<int64_t> coefficientPerLevel(depth, 0);

    for (affine::AffineForOp loop : band) {
        loop.walk([&](Operation *operation) {
            if (auto loadOperation = dyn_cast<affine::AffineLoadOp>(operation)) {
                visitedBuffers.insert(loadOperation.getMemref());
            } else if (auto storeOperation = dyn_cast<affine::AffineStoreOp>(operation)) {
                visitedBuffers.insert(storeOperation.getMemref());
            }
        });
    }

    for (Value buffer : visitedBuffers) {
        if (auto type = buffer.getType().dyn_cast<MemRefType>()) {
            unsigned elementBitWidth = type.getElementTypeBitWidth();
            int64_t elementBytes = std::max<unsigned>(1, elementBitWidth) / 8;
            if (elementBytes == 0) {
                elementBytes = 1;
            }
            int64_t rank = type.getRank();

            // 简化线性近似: 将 rank * elementBytes 分摊到每一层.
            int64_t perLevel = elementBytes * std::max<int64_t>(1, rank);
            for (size_t i = 0; i < depth; ++i) {
                coefficientPerLevel[i] += perLevel;
            }
            baseBytes += elementBytes;
        }
    }
    return coefficientPerLevel;
}

static LogicalResult solveOneBandAndWriteAttribute(func::FuncOp functionOperation,
                                                   const SmallVector<affine::AffineForOp, 8> &band,
                                                   int64_t vectorWidth, bool enforceVectorMultiple,
                                                   int64_t sharedMemoryLimitBytes,
                                                   int64_t fallbackExtentUpperBound) {
    using operations_research::Domain;
    using operations_research::sat::CpModelBuilder;
    using operations_research::sat::CpSolverResponse;
    using operations_research::sat::IntVar;
    using operations_research::sat::LinearExpr;
    using operations_research::sat::SolveCpModel;

    size_t depth = band.size();
    if (depth == 0) {
        return success();
    }

    // 1) 层上界: 属性优先, 否则常量解析或回退.
    std::vector<int64_t> extentUpperBound =
        getLayerExtentsFromAttributesOrFallback(band, fallbackExtentUpperBound);

    // 2) 依赖提示: 属性优先, 给出每层的软硬约束信息.
    std::vector<int64_t> layerUpperCap;
    std::vector<int64_t> layerWeights;
    getDependHintsFromAttributesOrFallback(band, layerUpperCap, layerWeights,
                                           /*defaultUpperCap=*/fallbackExtentUpperBound,
                                           /*innermostBonus=*/8);

    // 3) 共享内存线性系数: 属性优先, 否则回退估计.
    int64_t sharedMemoryBaseBytes = 0;
    std::vector<int64_t> sharedMemoryCoefficient =
        getSharedMemoryCoefficientsOrFallback(band, sharedMemoryBaseBytes);

    // 4) 构建整数线性模型.
    CpModelBuilder model;

    std::vector<IntVar> tileSizeVariables;
    tileSizeVariables.reserve(depth);
    for (size_t i = 0; i < depth; ++i) {
        int64_t lower = 1;
        int64_t upper = std::min<int64_t>(std::max<int64_t>(1, extentUpperBound[i]),
                                          std::max<int64_t>(1, layerUpperCap[i]));
        IntVar variable = model.NewIntVar(operations_research::Domain(lower, upper));
        tileSizeVariables.push_back(variable);
    }

    // 资源约束: 共享内存上界(线性).
    LinearExpr sharedUsage = LinearExpr(sharedMemoryBaseBytes);
    for (size_t i = 0; i < depth; ++i) {
        sharedUsage = sharedUsage + sharedMemoryCoefficient[i] * tileSizeVariables[i];
    }
    if (sharedMemoryLimitBytes > 0) {
        model.AddLessOrEqual(sharedUsage, sharedMemoryLimitBytes);
    }

    // 向量宽度约束: 最内层必须为向量宽度的整数倍(可选).
    if (enforceVectorMultiple && depth > 0 && vectorWidth > 1) {
        int64_t innermostUpper = extentUpperBound[depth - 1];
        int64_t maximumQuotient = std::max<int64_t>(1, innermostUpper / vectorWidth);
        auto quotient = model.NewIntVar(operations_research::Domain(1, maximumQuotient));
        model.AddEquality(tileSizeVariables[depth - 1], quotient * vectorWidth);
    }

    // 目标函数: 最大化 sum_i weight_i * T_i.
    LinearExpr objective(0);
    for (size_t i = 0; i < depth; ++i) {
        objective = objective + static_cast<int64_t>(layerWeights[i]) * tileSizeVariables[i];
    }
    model.Maximize(objective);

    // 5) 求解.
    operations_research::sat::Model solverModel;
    CpSolverResponse response = SolveCpModel(model.Build(), &solverModel);

    if (response.status() != operations_research::sat::CpSolverStatus::OPTIMAL &&
        response.status() != operations_research::sat::CpSolverStatus::FEASIBLE) {
        functionOperation.emitRemark() << "ILP 求解未找到可行解, 跳过该 band.";
        return failure();
    }

    // 6) 写回属性.
    SmallVector<int64_t, 8> tileSizes;
    for (size_t i = 0; i < depth; ++i) {
        int64_t value =
            operations_research::sat::SolutionIntegerValue(response, tileSizeVariables[i]);
        if (value < 1) {
            value = 1;
        }
        tileSizes.push_back(value);
    }

    MLIRContext *context = band.front()->getContext();
    SmallVector<Attribute, 8> elementAttributes;
    for (int64_t value : tileSizes) {
        elementAttributes.push_back(IntegerAttr::get(IntegerType::get(context, 64), value));
    }
    ArrayAttr array = ArrayAttr::get(context, elementAttributes);

    affine::AffineForOp outerMost = band.front();
    outerMost->setAttr("tile.sizes", array);
    outerMost.emitRemark() << "已选择分块大小并写回属性 tile.sizes.";

    return success();
}

class GpuTileILPSelectionPass
    : public PassWrapper<GpuTileILPSelectionPass, OperationPass<func::FuncOp>> {
   public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GpuTileILPSelectionPass)

    GpuTileILPSelectionPass() = default;
    GpuTileILPSelectionPass(const GpuTileILPSelectionPass &other) {}

    StringRef getArgument() const override { return "gpu-ilp-select-tiles"; }

    StringRef getDescription() const override {
        return "融合多面体属性提示, 用整数线性规划为 affine "
               "循环带选择分块大小并写回 tile.sizes.";
    }

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<affine::AffineDialect>();
        registry.insert<func::FuncDialect>();
    }

    Option<int64_t> optionVectorWidth{*this, "vector-width",
                                      llvm::cl::desc("最内层分块需要满足的向量宽度倍数, 默认 8."),
                                      llvm::cl::init(8)};

    Option<bool> optionEnforceVectorMultiple{
        *this, "enforce-vector-multiple",
        llvm::cl::desc("是否强制最内层分块为向量宽度的整数倍, 默认 true."), llvm::cl::init(true)};

    Option<int64_t> optionSharedMemoryLimitBytes{
        *this, "shared-memory-limit-bytes",
        llvm::cl::desc("共享内存总上界字节数用于线性资源约束, 默认 98304."),
        llvm::cl::init(96 * 1024)};

    Option<int64_t> optionFallbackExtentUpperBound{
        *this, "fallback-extent-upper-bound",
        llvm::cl::desc("当无法解析常量迭代次数时的回退上界, 默认 1024."), llvm::cl::init(1024)};

    void runOnOperation() override {
        SmallVector<affine::AffineForOp, 8> outerLoops;
        getOperation().walk([&](affine::AffineForOp loop) {
            if (!loop->getParentOfType<affine::AffineForOp>()) {
                outerLoops.push_back(loop);
            }
        });

        for (affine::AffineForOp outer : outerLoops) {
            SmallVector<affine::AffineForOp, 8> band;
            collectPerfectNestedBand(outer, band);
            if (band.empty()) {
                continue;
            }
            if (outer->hasAttr("tile.sizes")) {
                outer.emitRemark() << "已存在 tile.sizes 属性, 跳过该 band.";
                continue;
            }

            LogicalResult result = solveOneBandAndWriteAttribute(
                getOperation(), band, optionVectorWidth, optionEnforceVectorMultiple,
                optionSharedMemoryLimitBytes, optionFallbackExtentUpperBound);

            if (failed(result)) {
                continue;
            }
        }
    }
};

}  // end anonymous namespace

std::unique_ptr<Pass> createGpuTileILPSelectionPass() {
    return std::make_unique<GpuTileILPSelectionPass>();
}

extern "C" ::mlir::PassPluginLibraryInfo mlirGetPassPluginInfo() {
    return {MLIR_PLUGIN_API_VERSION,
            /*pluginName=*/"GpuTileILPSelection",
            /*pluginVersion=*/"v0.2-poly-attrs",
            /*registerPassPipeline=*/
            []() {
                static ::mlir::PassRegistration<GpuTileILPSelectionPass> reg;
                ::mlir::PassPipelineRegistration<>(
                    "gpu-ilp-select-tiles",
                    "Polyhedra-informed ILP tile size selection, writes attribute "
                    "tile.sizes.",
                    [](::mlir::OpPassManager &pm) { pm.addPass(createGpuTileILPSelectionPass()); });
            }};
}
