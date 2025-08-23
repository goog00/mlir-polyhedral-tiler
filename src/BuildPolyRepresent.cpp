// 目的:在“循环表示（Representation）”这一步，扫描 affine 循环，
//      以每个 perfect-nested 循环带（band）为单位，构造其迭代域的
//      Presburger FlatLinearValueConstraints
//      后续步骤（依赖分析/调度/ILP）可基于此继续实现。

//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

// Presburger 约束
#include "mlir/Analysis/FlatLinearValueConstraints.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Tools/Plugins/PassPlugin.h"

// LLVM 容器
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {

/// 自实现:从 root 开始向内，若循环体(不含 terminator)只有一个 op 且是
/// affine.for，则继续收集。
static void collectPerfectBand(affine::AffineForOp root,
                               SmallVectorImpl<affine::AffineForOp> &band) {
    band.clear();
    if (!root) return;

    affine::AffineForOp cur = root;
    while (true) {
        band.push_back(cur);

        // getBody() 在 18.x 返回 Block*
        Block *body = cur.getBody();
        affine::AffineForOp inner = nullptr;
        unsigned count = 0;

        // 迭代 block 内所有 op;手动跳过 terminator(affine.yield)
        for (Operation &op : *body) {
            if (isa<affine::AffineYieldOp>(op)) continue;
            ++count;
            if (count == 1)
                if (auto next = dyn_cast<affine::AffineForOp>(op)) inner = next;
        }

        if (count == 1 && inner)
            cur = inner;  // perfect 嵌套，继续向内
        else
            break;  // 非 perfect，结束
    }
}

/// 为一个 band 构建 FlatLinearValueConstraints:
/// - band 每层 iv 作为 dim;
/// - 上下界 map 的非 iv 操作数作为 symbol;
/// - computeAlignedMap 对齐后，用 addBound(LB/UB) 加入约束。
static LogicalResult buildFLVCForBand(ArrayRef<affine::AffineForOp> band,
                                      FlatLinearValueConstraints &vlc) {
    if (band.empty()) {
        return failure();
    }

    vlc = FlatLinearValueConstraints();

    // dims = 所有 iv(外到内保序)
    SmallVector<Value, 8> dimIVs;
    dimIVs.reserve(band.size());
    for (auto forOp : band) {
        dimIVs.push_back(forOp.getInductionVar());
    }

    vlc.appendDimVar(ValueRange(dimIVs));

    // symbols = 所有上下界 map 中的 非 iv 操作数
    llvm::DenseSet<Value> ivSet;
    ivSet.reserve(dimIVs.size());
    for (Value v : dimIVs) {
        ivSet.insert(v);
    }

    llvm::DenseSet<Value> symSet;
    auto collectSymbol = [&](ValueRange ops) {
        for (Value v : ops) {
            if (!v) {
                continue;
            }

            if (ivSet.contains(v)) {
                continue;
            }

            symSet.insert(v);
        }
    };
    for (auto forOp : band) {
        collectSymbol(forOp.getLowerBoundOperands());
        collectSymbol(forOp.getUpperBoundOperands());
    }

    SmallVector<Value, 8> symVals;
    symVals.reserve(symSet.size());
    for (Value v : symSet) symVals.push_back(v);
    if (!symVals.empty()) vlc.appendSymbolVar(ValueRange(symVals));

    //  加上下界
    for (unsigned pos = 0; pos < band.size(); ++pos) {
        affine::AffineForOp forOp = band[pos];

        // LB(默认闭)
        AffineMap lbMap = forOp.getLowerBoundMap();
        AffineMap lbAligned = vlc.computeAlignedMap(lbMap, forOp.getLowerBoundOperands());
        (void)vlc.addBound(presburger::BoundType::LB, /*pos=*/pos, lbAligned);

        // UB(默认开)
        AffineMap ubMap = forOp.getUpperBoundMap();
        AffineMap ubAligned = vlc.computeAlignedMap(ubMap, forOp.getUpperBoundOperands());
        (void)vlc.addBound(presburger::BoundType::UB, /*pos=*/pos, ubAligned);

        // step:此处先不显式编码整除关系;必要时用 local 等式引入。
        (void)forOp.getStep();
    }

    return success();
}

/// Pass:打印每个 band 的 FLVC 概要与矩阵
struct BuildPolyRepresentPass
    : public PassWrapper<BuildPolyRepresentPass, OperationPass<func::FuncOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BuildPolyRepresentPass)

    void runOnOperation() override {
        func::FuncOp func = getOperation();

        SmallVector<affine::AffineForOp, 8> roots;
        func.walk([&](affine::AffineForOp forOp) {
            if (!forOp->getParentOfType<affine::AffineForOp>()) roots.push_back(forOp);
        });

        if (roots.empty()) {
            llvm::errs() << " 未发现 affine.for;请先做 "
                            "-one-shot-bufferize 与 -convert-linalg-to-affine-loops。\n";
            return;
        }

        int bandId = 0;
        for (auto root : roots) {
            SmallVector<affine::AffineForOp, 8> band;
            collectPerfectBand(root, band);

            llvm::errs() << "\n 循环带 #" << bandId++ << "(深度 " << band.size() << ")\n";

            for (unsigned i = 0; i < band.size(); ++i) {
                auto forOp = band[i];
                llvm::errs() << "  L" << i << ": for %iv in [" << forOp.getLowerBoundMap() << ", "
                             << forOp.getUpperBoundMap() << ") step " << forOp.getStep() << "\n";
            }

            FlatLinearValueConstraints vlc;
            if (failed(buildFLVCForBand(band, vlc))) {
                llvm::errs() << " 构建约束失败。\n";
                continue;
            }

            // 使用 getNum*Vars()
            llvm::errs() << "  约束概要:dims=" << vlc.getNumDimVars()
                         << ", symbols=" << vlc.getNumSymbolVars()
                         << ", locals=" << vlc.getNumLocalVars()
                         << ", 等式=" << vlc.getNumEqualities()
                         << ", 不等式=" << vlc.getNumInequalities() << "\n";

            llvm::errs() << "  详细矩阵(行=约束;列=dim/sym/local + 常数):\n";
            vlc.dump();

            // 打印 dim ↔ SSA 名
            auto dims = vlc.getMaybeValues(mlir::presburger::VarKind::SetDim);
            for (unsigned i = 0; i < dims.size(); ++i) {
                if (dims[i]) {
                    llvm::errs() << "  d" << i << " <- " << *dims[i] << "\n";
                }
            }
        }
    }

    StringRef getArgument() const final { return "poly-build-repr"; }
    StringRef getDescription() const final {
        return "构造并打印 affine 循环带(band)的 "
               "FlatLinearValueConstraints(llvmorg-18.1.8)。";
    }
};

}  // namespace

//=== 插件入口:以 pass 和 pipeline 的形式注册 ============================//

static std::unique_ptr<mlir::Pass> createBuildPolyRepresentPass() {
    return std::make_unique<BuildPolyRepresentPass>();
}

extern "C" ::mlir::PassPluginLibraryInfo mlirGetPassPluginInfo() {
    return {MLIR_PLUGIN_API_VERSION,
            /*pluginName=*/"BuildPolyRepresent",
            /*pluginVersion=*/"v0.4-no-looputils",
            /*registerPassPipeline=*/[]() {
                // 注册命名 Pass(等价于旧的 static PassRegistration)
                ::mlir::PassRegistration<BuildPolyRepresentPass>();

                // 注册一个 pipeline:-poly-build-repr
                ::mlir::PassPipelineRegistration<>(
                    "poly-build-repr",
                    "Build and print FlatLinearValueConstraints for each affine "
                    "perfect-nested loop band (llvmorg-18.1.8).",
                    [](::mlir::OpPassManager &pm) { pm.addPass(createBuildPolyRepresentPass()); });
            }};
}
