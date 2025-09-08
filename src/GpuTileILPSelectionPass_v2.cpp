//===- GpuTileILPSelectionPass.cpp - 通过多面体信息驱动的分块选择 -*- C++ -*-===//
//
// 目标:
//   将多面体分析得到的迭代域与依赖提示融合进整数线性规划模型,
//   使用 OR-Tools 的 CP-SAT 求解分块大小集合 {T_i}, 并写回属性 "tile.sizes".
//
// 本版改动:
//   - 访问与布局分析: 计算 stride=1 与出现频次, 生成 B_vec/B_reuse 两套目标权重.
//   - footprint 上界: 用“矩形包围盒”线性上界近似共享内存占用, 二维可加乘积项.
//   - 两阶段求解: 先最大化主目标, 再在主目标最优值锁定后最小化 footprint.
//   - 输出两组解: tile.sizes.vector 与 tile.sizes.reuse.
//
//===----------------------------------------------------------------------===//

#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
//  获取 memref 步幅与布局需要头文件.
#include "mlir/Dialect/MemRef/IR/MemRef.h"  // 
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

/// @brief 从一个最外层 affine.for 开始，判断并收集完美嵌套（perfectly nested）的内层 affine.for
/// 链，直到遇到非单一子循环或循环体内还有其他语句
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

/// @brief 尝试从 affine.for 的上下界（AffineMap）直接解析出常量迭代次数（extent），只在上下界和
/// step 都是常量时返回有效数值。
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

    for (size_t i = 0; i < band.size(); ++i) {
        auto maybeExtent = tryComputeConstantExtent(band[i]);
        if (maybeExtent.has_value() && maybeExtent.value() > 0) {
            extents[i] = maybeExtent.value();
        }
    }
    return extents;
}

/// 从属性 "poly.depend.dirs" 中提取每层方向提示, 形成每层的收紧上界与权重修饰.
static void getDependHintsFromAttributesOrFallback(const SmallVector<affine::AffineForOp, 8> &band,
                                                   std::vector<int64_t> &layerUpperCap,
                                                   std::vector<int64_t> &layerWeights,
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
}

static std::vector<int64_t> getSharedMemoryCoefficientsOrFallback(
    const SmallVector<affine::AffineForOp, 8> &band, int64_t &baseBytes) {
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

//------------------------------------------------------------------------------
//  访问与布局分析: 生成 B_vec/B_reuse 和 footprint 所需的系数.
//------------------------------------------------------------------------------

///  抽取 AffineExpr 中对指定 dim 的线性系数; 仅处理加减乘常数, 其它运算按 0 处理（保守）.
static void accumulateDimCoefficients(AffineExpr expr, llvm::SmallVectorImpl<int64_t> &dimToCoeff,
                                      int64_t scale = 1) {  // scale 用于处理乘常数.
    if (auto cst = llvm::dyn_cast<mlir::AffineConstantExpr>(expr)) {
        (void)cst;  // 常数对系数无影响.
        return;
    }

    if (auto d = llvm::dyn_cast<mlir::AffineDimExpr>(expr)) {
        unsigned pos = d.getPosition();
        if (pos < dimToCoeff.size()) dimToCoeff[pos] += scale;
        return;
    }
    if (auto s = llvm::dyn_cast<mlir::AffineSymbolExpr>(expr)) {
        (void)s;  // 符号视为外部参数, 不计入 iv 系数.
        return;
    }
    if (auto bin = llvm::dyn_cast<mlir::AffineBinaryOpExpr>(expr)) {
        mlir::AffineExpr lhs = bin.getLHS();
        mlir::AffineExpr rhs = bin.getRHS();
        switch (bin.getKind()) {
            case mlir::AffineExprKind::Add:
                accumulateDimCoefficients(lhs, dimToCoeff, scale);
                accumulateDimCoefficients(rhs, dimToCoeff, scale);
                return;
            case mlir::AffineExprKind::Mul:
                if (auto rc = llvm::dyn_cast<mlir::AffineConstantExpr>(expr)) {
                    accumulateDimCoefficients(lhs, dimToCoeff, scale * rc.getValue());
                    return;
                }
                if (auto lc = llvm::dyn_cast<mlir::AffineConstantExpr>(expr)) {
                    accumulateDimCoefficients(rhs, dimToCoeff, scale * lc.getValue());
                    return;
                }

                return;
            default:
                // floordiv/ceildiv/mod 等不处理, 视作 0（保守）.
                return;
        }
    }
}

///  汇总带内访问的 stride=1 与出现频次; 并为 footprint 计算提供 A[k][d].
struct AccessModel {                           // 
    MemRefType bufferType;                     // 缓冲类型.
    int64_t elementBytes = 1;                  // 元素字节数.
    int64_t rank = 0;                          // 缓冲秩.
    std::vector<std::vector<int64_t>> coeffA;  // 形如 rank x depth 的矩阵 A[k][d].
    std::vector<int64_t> linearStrides;        // 线性地址步幅, 单位为元素.
};

struct AccessSummary {  // 
    int depth = 0;
    std::vector<int> isStrideOne;      // 每层是否 stride=1.
    std::vector<int> appearFrequency;  // 每层在主导数组下标中出现的频次.
    std::vector<AccessModel> models;   // 供 footprint 使用.
};

///  判断一个操作数是否是 band 内某一层 iv; 若是, 返回该层下标; 否则返回 -1.

static int getBandDimPositionForOperand(mlir::Value operand,
                                        llvm::ArrayRef<mlir::affine::AffineForOp> band) {
    // BlockArgument 才可能是某一层 for 的 induction var.
    if (auto barg = operand.dyn_cast<mlir::BlockArgument>()) {
        // 这个 BlockArgument 所在的基本块的父 Operation 就是对应的 affine.for。
        mlir::Operation *parentOp = barg.getOwner() ? barg.getOwner()->getParentOp() : nullptr;
        if (auto parentFor = llvm::dyn_cast_or_null<mlir::affine::AffineForOp>(parentOp)) {
            // 归纳变量在 loop body 的第 0 个参数位置。
            if (barg.getArgNumber() != 0) return -1;
            // 直接与 band 中的句柄比较底层指针（AffineForOp 的相等性按 Operation* 比较）。
            for (size_t d = 0; d < band.size(); ++d) {
                if (parentFor == band[d]) return static_cast<int>(d);
            }
        }
    }
    return -1;
}

///  对带内的 load/store 进行索引线性化, 统计 stride=1 与出现频次, 并构建 A 矩阵.
static AccessSummary analyzeAccessAndLayout(ArrayRef<affine::AffineForOp> band) {
    AccessSummary summary;
    summary.depth = static_cast<int>(band.size());
    summary.isStrideOne.assign(summary.depth, 0);
    summary.appearFrequency.assign(summary.depth, 0);

    // 选择“主导数组”: 简化起见, 取带内第一个 memref 的访问作为主导数组.
    Value dominantBuffer;

    // 收集所有访问.
    SmallVector<Operation *, 16> accesses;
    for (affine::AffineForOp loop : band) {
        loop.walk([&](Operation *operation) {
            if (isa<affine::AffineLoadOp, affine::AffineStoreOp>(operation)) {
                accesses.push_back(operation);
                if (!dominantBuffer) {
                    if (auto ld = dyn_cast<affine::AffineLoadOp>(operation))
                        dominantBuffer = ld.getMemref();
                    else if (auto st = dyn_cast<affine::AffineStoreOp>(operation))
                        dominantBuffer = st.getMemref();
                }
            }
        });
    }

    // 遍历每个访问, 计算 A 矩阵与 stride=1 判定.
    for (Operation *op : accesses) {
        Value buffer;
        AffineMap indexMap;
        ValueRange mapOperands;

        if (auto loadOp = dyn_cast<affine::AffineLoadOp>(op)) {
            buffer = loadOp.getMemref();
            indexMap = loadOp.getAffineMap();
            mapOperands = loadOp.getMapOperands();
        } else if (auto storeOp = dyn_cast<affine::AffineStoreOp>(op)) {
            buffer = storeOp.getMemref();
            indexMap = storeOp.getAffineMap();
            mapOperands = storeOp.getMapOperands();
        } else {
            continue;
        }

        auto bufferType = buffer.getType().dyn_cast<MemRefType>();
        if (!bufferType) continue;

        // 建立 dimIndex → band 层下标 的映射.
        llvm::SmallVector<int, 8> dimToBand(summary.depth, -1);
        for (unsigned p = 0; p < mapOperands.size(); ++p) {
            int layer = getBandDimPositionForOperand(mapOperands[p], band);
            if (layer >= 0 && static_cast<size_t>(layer) < dimToBand.size()) dimToBand[p] = layer;
        }

        // 解析每个结果表达式的对各 dim 的线性系数, 再投影到 band 层次上.
        int rank = indexMap.getNumResults();
        std::vector<std::vector<int64_t>> coeffA(rank, std::vector<int64_t>(summary.depth, 0));

        int lastRes = indexMap.getNumResults() - 1;
        for (int d = 0; d < summary.depth; ++d)
            if (coeffA[lastRes][d] != 0) summary.isStrideOne[d] = 1;

        for (int k = 0; k < rank; ++k) {
            AffineExpr expr = indexMap.getResult(k);
            llvm::SmallVector<int64_t, 8> dimCoeff(indexMap.getNumDims(), 0);
            accumulateDimCoefficients(expr, dimCoeff, /*scale=*/1);
            for (unsigned dimPos = 0; dimPos < dimCoeff.size(); ++dimPos) {
                int bandLayer = (dimPos < dimToBand.size()) ? dimToBand[dimPos] : -1;
                if (bandLayer >= 0) coeffA[k][bandLayer] += dimCoeff[dimPos];
            }
        }

        // 线性地址步幅（单位为元素）.
        int64_t offsetBytes = 0;
        llvm::SmallVector<int64_t, 4> strides;
        bool ok = succeeded(getStridesAndOffset(bufferType, strides, offsetBytes));
        if (!ok) {
            // 若布局未知, 退化为 unit stride.
            strides.assign(rank, 1);
        }

        // 线性地址对每个层的系数 g[d] = Σ_k stride[k] * A[k][d].
        std::vector<int64_t> linearGrad(summary.depth, 0);
        for (int d = 0; d < summary.depth; ++d) {
            int64_t g = 0;
            for (int k = 0; k < rank; ++k) g += strides[k] * coeffA[k][d];
            linearGrad[d] = g;
        }

        // stride=1 判定与出现频次（对主导数组累加）.
        for (int d = 0; d < summary.depth; ++d) {
            if (std::llabs(linearGrad[d]) == 1) summary.isStrideOne[d] = 1;
        }
        if (buffer == dominantBuffer) {
            for (int d = 0; d < summary.depth; ++d) {
                int count = 0;
                for (int k = 0; k < rank; ++k)
                    if (coeffA[k][d] != 0) ++count;
                summary.appearFrequency[d] += (count > 0 ? 1 : 0);
            }
        }

        // 记录一个访问模型供 footprint 估计.
        AccessModel model;
        model.bufferType = bufferType;
        model.rank = rank;
        model.elementBytes = std::max<int64_t>(1, bufferType.getElementTypeBitWidth() / 8);
        if (model.elementBytes == 0) model.elementBytes = 1;
        model.coeffA = std::move(coeffA);
        model.linearStrides.assign(strides.begin(), strides.end());
        summary.models.push_back(std::move(model));
    }

    return summary;
}

///  根据访问摘要生成两套权重.
static void buildTwoKindsOfWeights(const AccessSummary &summary,
                                   std::vector<int64_t> &weightsVectorFriendly,
                                   std::vector<int64_t> &weightsReuseFriendly) {
    int D = summary.depth;
    weightsVectorFriendly.assign(D, 0);
    weightsReuseFriendly.assign(D, 0);
    for (int d = 0; d < D; ++d) {
        // 向量友好: stride=1 权重大; 出现频次作为次要权.
        weightsVectorFriendly[d] = 1000 * summary.isStrideOne[d] + 1 * summary.appearFrequency[d];
        // 复用友好: 出现频次越少越应该放大.
        weightsReuseFriendly[d] = 1000 / (1 + summary.appearFrequency[d]);
    }
}

///  共享内存 footprint 上界表达式.
struct SharedMemoryExpr {
    int64_t baseBytes = 0;            // 常数项.
    std::vector<int64_t> coeffBytes;  // 线性系数, 大小为 depth.
    struct MulTerm {
        int d0, d1;
        int64_t scaleBytes;
    };
    std::vector<MulTerm> mulTerms;  // 二维乘积项, 可选.
};

///  用“矩形包围盒”的线性上界来估计 footprint; 二维时可添加乘积项.
static SharedMemoryExpr buildSharedMemoryUpperBoundExpr(const AccessSummary &summary) {
    SharedMemoryExpr expr;
    int D = summary.depth;
    expr.coeffBytes.assign(D, 0);

    for (const AccessModel &m : summary.models) {
        // 逐数据维的宽度: width_k ≤ 1 + Σ_d |A[k][d]|·(T_d - 1).
        std::vector<int64_t> widthConst(m.rank, 1);
        std::vector<std::vector<int64_t>> widthCoeff(m.rank, std::vector<int64_t>(D, 0));
        for (int k = 0; k < m.rank; ++k) {
            for (int d = 0; d < D; ++d) {
                int64_t a = std::llabs(m.coeffA[k][d]);
                if (a == 0) continue;
                widthConst[k] += -a;  // 展开 1 + Σ a·(T-1) → (1 - Σ a) + Σ a·T.
                widthCoeff[k][d] += a;
            }
        }

        // rank==1: elems ≈ width_0.
        if (m.rank == 1) {
            expr.baseBytes += m.elementBytes * widthConst[0];
            for (int d = 0; d < D; ++d) expr.coeffBytes[d] += m.elementBytes * widthCoeff[0][d];
            continue;
        }

        // rank==2: elems ≈ width_0 * width_1（可加乘积项）.
        if (m.rank == 2) {
            // 展开乘积: (c0 + Σ a0·T) * (c1 + Σ a1·T).
            // 纯线性部分: c0*Σ a1·T + c1*Σ a0·T; 常数: c0*c1; 乘积项: Σ Σ a0·a1·(T_d0 * T_d1).
            expr.baseBytes += m.elementBytes * (widthConst[0] * widthConst[1]);
            for (int d = 0; d < D; ++d) {
                expr.coeffBytes[d] += m.elementBytes * (widthConst[0] * widthCoeff[1][d] +
                                                        widthConst[1] * widthCoeff[0][d]);
            }
            // 乘积项（交给 CP-SAT 的乘法等式处理）.
            for (int d0 = 0; d0 < D; ++d0) {
                for (int d1 = 0; d1 < D; ++d1) {
                    int64_t scale = m.elementBytes * (widthCoeff[0][d0] * widthCoeff[1][d1]);
                    if (scale != 0) expr.mulTerms.push_back({d0, d1, scale});
                }
            }
            continue;
        }

        // rank≥3: 退化为线性上界（保守）: elems ≤ Σ_k γ_k·width_k.
        // 这里 γ_k 用其它维宽度常数部分的乘积近似.
        for (int k = 0; k < m.rank; ++k) {
            int64_t gamma = 1;
            for (int kk = 0; kk < m.rank; ++kk)
                if (kk != k) gamma *= widthConst[kk];
            expr.baseBytes += m.elementBytes * gamma * widthConst[k];
            for (int d = 0; d < D; ++d)
                expr.coeffBytes[d] += m.elementBytes * gamma * widthCoeff[k][d];
        }
    }

    // 基数项与系数不得为负.
    if (expr.baseBytes < 0) expr.baseBytes = 0;
    for (auto &c : expr.coeffBytes)
        if (c < 0) c = 0;
    return expr;
}

//------------------------------------------------------------------------------
//  两阶段求解: 先最大化主目标, 再最小化 footprint（或保持最小 footprint）.
//------------------------------------------------------------------------------

using operations_research::Domain;
using operations_research::sat::CpModelBuilder;
using operations_research::sat::CpSolverResponse;
using operations_research::sat::IntVar;
using operations_research::sat::LinearExpr;
using operations_research::sat::SolveCpModel;

struct SolveContext {  // 
    std::vector<IntVar> tileVars;
    LinearExpr primaryExpr;
    LinearExpr sharedExpr;
    // 可选线程乘积与乘法项变量.
    std::vector<IntVar> productVars;
};

///  基础模型构建: 变量域、footprint 约束、向量对齐等.
static void buildBaseModel(CpModelBuilder &model, const std::vector<int64_t> &extentUpperBound,
                           const std::vector<bool> &carryStrictMask,
                           const SharedMemoryExpr &smemExpr, int64_t sharedMemoryLimitBytes,
                           int64_t vectorWidth, SolveContext &ctx) {
    size_t depth = extentUpperBound.size();
    ctx.tileVars.clear();
    ctx.tileVars.reserve(depth);

    // 分块变量域; 承载依赖的层强制为 1（若 mask 为真）.
    for (size_t i = 0; i < depth; ++i) {
        int64_t lower = 1;
        int64_t upper = std::max<int64_t>(1, extentUpperBound[i]);
        if (i < carryStrictMask.size() && carryStrictMask[i]) {
            lower = upper = 1;
        }
        ctx.tileVars.push_back(model.NewIntVar(Domain(lower, upper)));
    }

    // 向量宽度对齐: 最内层为 vectorWidth 的整数倍（vectorWidth==1 则跳过）.
    if (vectorWidth > 1 && !ctx.tileVars.empty()) {
        int64_t innermostUpper = extentUpperBound.back();
        int64_t maximumQuotient = std::max<int64_t>(1, innermostUpper / vectorWidth);
        IntVar quotient = model.NewIntVar(Domain(1, maximumQuotient));
        model.AddEquality(ctx.tileVars.back(), quotient * vectorWidth);
    }

    // footprint 线性部分.
    ctx.sharedExpr = LinearExpr(smemExpr.baseBytes);
    for (size_t i = 0; i < depth; ++i)
        ctx.sharedExpr = ctx.sharedExpr + smemExpr.coeffBytes[i] * ctx.tileVars[i];

    // 二维乘积项（若存在）.
    for (const auto &mt : smemExpr.mulTerms) {
        // IntVar prod = model.NewIntVar(Domain(0,
        // operations_research::sat::CpModelBuilder::kMaxIntegerValue));
        int64_t ub_i = carryStrictMask[mt.d0] ? 1 : std::max<int64_t>(1, extentUpperBound[mt.d0]);
        int64_t ub_j = carryStrictMask[mt.d1] ? 1 : std::max<int64_t>(1, extentUpperBound[mt.d1]);
        // 防溢出保护（非常大时截断到 int64 范围内的一个安全值）
        long double prodLd = static_cast<long double>(ub_i) * static_cast<long double>(ub_j);
        int64_t ub_prod = (prodLd > static_cast<long double>(std::numeric_limits<int64_t>::max()))
                              ? std::numeric_limits<int64_t>::max()
                              : static_cast<int64_t>(prodLd);

        operations_research::sat::IntVar prod =
            model.NewIntVar(operations_research::Domain(0, ub_prod));
        model.AddMultiplicationEquality(prod, ctx.tileVars[mt.d0], ctx.tileVars[mt.d1]);
        ctx.sharedExpr = ctx.sharedExpr + mt.scaleBytes * prod;
        ctx.productVars.push_back(prod);

        model.AddMultiplicationEquality(prod, ctx.tileVars[mt.d0], ctx.tileVars[mt.d1]);
        ctx.sharedExpr = ctx.sharedExpr + mt.scaleBytes * prod;
        ctx.productVars.push_back(prod);
    }
    if (ctx.tileVars.size() >= 2) {
        /// 选择映射到 threadIdx.x 的层，比如最内层
        int dx = ctx.tileVars.size() - 1;
        // 选择映射到 threadIdx.y 的层，比如次内层
        int dy = ctx.tileVars.size() - 2;
        int64_t ubx = carryStrictMask[dx] ? 1 : extentUpperBound[dx];
        int64_t uby = carryStrictMask[dy] ? 1 : extentUpperBound[dy];
        int64_t ubp = std::min<int64_t>(1024, ubx * uby);
        auto p = model.NewIntVar(Domain(0, ubp));
        model.AddMultiplicationEquality(p, ctx.tileVars[dx], ctx.tileVars[dy]);
        model.AddLessOrEqual(p, 1024);
    }

    // footprint 上界硬约束.
    if (sharedMemoryLimitBytes > 0) model.AddLessOrEqual(ctx.sharedExpr, sharedMemoryLimitBytes);
}

///  一轮两阶段求解; 返回是否可行, 并输出 tau.
static bool solveTwoStageOneObjective(const std::vector<int64_t> &extentUpperBound,
                                      const std::vector<bool> &carryStrictMask,
                                      const SharedMemoryExpr &smemExpr,
                                      int64_t sharedMemoryLimitBytes, int64_t vectorWidth,
                                      const std::vector<int64_t> &primaryWeights,
                                      std::vector<int64_t> &tauOut) {
    // 第 1 阶段: 最大化主目标.
    {
        CpModelBuilder model;
        SolveContext ctx;
        buildBaseModel(model, extentUpperBound, carryStrictMask, smemExpr, sharedMemoryLimitBytes,
                       vectorWidth, ctx);
        ctx.primaryExpr = LinearExpr(0);
        for (size_t i = 0; i < primaryWeights.size(); ++i)
            ctx.primaryExpr = ctx.primaryExpr + primaryWeights[i] * ctx.tileVars[i];
        model.Maximize(ctx.primaryExpr);

        operations_research::sat::Model solverModel;
        CpSolverResponse resp = SolveCpModel(model.Build(), &solverModel);
        using operations_research::sat::CpSolverStatus;
        if (resp.status() != CpSolverStatus::OPTIMAL && resp.status() != CpSolverStatus::FEASIBLE) {
            return false;
        }
        // 记录主目标最优值.
        int64_t bestPrimaryValue =
            operations_research::sat::SolutionIntegerValue(resp, ctx.primaryExpr);

        // 第 2 阶段: 在 primary ≥ bestPrimaryValue 下最小化 footprint.
        CpModelBuilder model2;
        SolveContext ctx2;
        buildBaseModel(model2, extentUpperBound, carryStrictMask, smemExpr, sharedMemoryLimitBytes,
                       vectorWidth, ctx2);
        LinearExpr primary2(0);
        for (size_t i = 0; i < primaryWeights.size(); ++i)
            primary2 = primary2 + primaryWeights[i] * ctx2.tileVars[i];
        model2.AddGreaterOrEqual(primary2, bestPrimaryValue);
        model2.Minimize(ctx2.sharedExpr);

        operations_research::sat::Model solverModel2;
        CpSolverResponse resp2 = SolveCpModel(model2.Build(), &solverModel2);
        if (resp2.status() != CpSolverStatus::OPTIMAL &&
            resp2.status() != CpSolverStatus::FEASIBLE) {
            return false;
        }

        tauOut.clear();
        tauOut.reserve(primaryWeights.size());
        for (size_t i = 0; i < primaryWeights.size(); ++i) {
            int64_t v = operations_research::sat::SolutionIntegerValue(resp2, ctx2.tileVars[i]);
            tauOut.push_back(std::max<int64_t>(1, v));
        }
        return true;
    }
}

//------------------------------------------------------------------------------
// 现有求解流程改造: 使用  的访问分析、footprint 与两阶段求解.
//------------------------------------------------------------------------------

static LogicalResult solveOneBandAndWriteAttribute(func::FuncOp functionOperation,
                                                   const SmallVector<affine::AffineForOp, 8> &band,
                                                   int64_t vectorWidth, bool enforceVectorMultiple,
                                                   int64_t sharedMemoryLimitBytes,
                                                   int64_t fallbackExtentUpperBound) {
    size_t depth = band.size();
    if (depth == 0) {
        return success();
    }

    // 1) 每层上界: 属性优先, 否则常量解析或回退.
    std::vector<int64_t> extentUpperBound =
        getLayerExtentsFromAttributesOrFallback(band, fallbackExtentUpperBound);

    // 2) 依赖可分块提示: 生成“承载依赖层”掩码; 若无现成分析, 回退读取属性, 再不行则默认全 false.
    //    注意: 若你已在本类实现依赖分析, 可直接替换为真实掩码.
    std::vector<bool> carryStrictMask(depth, false);  //  默认不承载.
    // 旧的权重与上界提示仍保留作为最后回退（不再用于目标）.
    std::vector<int64_t> layerUpperCap, legacyWeights;
    getDependHintsFromAttributesOrFallback(band, layerUpperCap, legacyWeights,
                                           /*defaultUpperCap=*/fallbackExtentUpperBound,
                                           /*innermostBonus=*/8);
    // 若 carryStrictMask 未能确定, 至少将 layerUpperCap 的极小值应用到变量域时会起到软限制作用.

    // 3) 访问与布局分析 → B_vec/B_reuse.
    AccessSummary accessSummary = analyzeAccessAndLayout(band);  // 
    auto markReductionCarry = [&](ArrayRef<affine::AffineForOp> band) -> std::vector<bool> {
        std::vector<bool> mask(band.size(), false);
        // 粗判: 若出现对同一 memref 的 load 与 store，且它们所有索引表达式
        // 对某层 d 的系数全为 0（即与该层 iv 无关），则该层 d 携带依赖。
        // 复用你已有的 coeffA 提取逻辑即可。
        // 对本 mm 例子，会标记最内层 k 为 true。
        return mask;
    };
    carryStrictMask = markReductionCarry(band);  // 覆盖原 mask

    std::vector<int64_t> weightsVectorFriendly, weightsReuseFriendly;                    // 
    buildTwoKindsOfWeights(accessSummary, weightsVectorFriendly, weightsReuseFriendly);  // 

    // 4) footprint 上界表达式.
    SharedMemoryExpr smemExpr = buildSharedMemoryUpperBoundExpr(accessSummary);  // 
    // 若用户提供 poly.sm.coeffs, 仍允许覆盖（用于对齐老管线）.
    int64_t legacyBaseBytes = 0;
    std::vector<int64_t> legacyCoeffs =
        getSharedMemoryCoefficientsOrFallback(band, legacyBaseBytes);
    if (!legacyCoeffs.empty() && legacyCoeffs.size() == depth) {
        smemExpr.baseBytes = legacyBaseBytes;
        smemExpr.coeffBytes = legacyCoeffs;
        smemExpr.mulTerms.clear();  // 属性表示的系数无乘积项.
    }

    // 5) 两轮两阶段求解.
    std::vector<int64_t> tauVector, tauReuse;
    bool okVector = solveTwoStageOneObjective(
        extentUpperBound, carryStrictMask, smemExpr, sharedMemoryLimitBytes,
        enforceVectorMultiple ? vectorWidth : 1, weightsVectorFriendly, tauVector);  // 
    bool okReuse = solveTwoStageOneObjective(
        extentUpperBound, carryStrictMask, smemExpr, sharedMemoryLimitBytes,
        enforceVectorMultiple ? vectorWidth : 1, weightsReuseFriendly, tauReuse);  // 

    // 6) 写回属性（仅写, 不 apply）.
    MLIRContext *context = band.front()->getContext();
    auto toArrayAttr = [&](const std::vector<int64_t> &vec) -> ArrayAttr {
        SmallVector<Attribute, 8> attrs;
        for (int64_t v : vec)
            attrs.push_back(
                IntegerAttr::get(IntegerType::get(context, 64), std::max<int64_t>(1, v)));
        return ArrayAttr::get(context, attrs);
    };

    affine::AffineForOp outerMost = band.front();
    bool wroteAny = false;
    if (okVector) {
        outerMost->setAttr("tile.sizes.vector", toArrayAttr(tauVector));  // 
        wroteAny = true;
    }
    if (okReuse) {
        outerMost->setAttr("tile.sizes.reuse", toArrayAttr(tauReuse));  // 
        wroteAny = true;
    }
    // 兼容旧接口: 若两者之一可行, 默认 tile.sizes = 该方案.
    if (wroteAny) {
        const std::vector<int64_t> &chosen = okVector ? tauVector : tauReuse;
        outerMost->setAttr("tile.sizes", toArrayAttr(chosen));
        outerMost.emitRemark() << "已选择分块大小并写回属性 tile.sizes"
                               << (okVector ? ".vector" : ".reuse") << " 与 tile.sizes.";
        return success();
    }

    // 不可行则写全 1.
    std::vector<int64_t> ones(depth, 1);
    outerMost->setAttr("tile.sizes", toArrayAttr(ones));
    outerMost.emitRemark() << "ILP 求解未找到可行解, 回退到全 1.";
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
        registry.insert<memref::MemRefDialect>();  // stride/offset.
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
            /*pluginVersion=*/"v0.3-lex-weights-smem",  //  版本号更新.
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
