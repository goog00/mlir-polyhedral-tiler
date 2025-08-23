//===- ApplyTileSizes.cpp - Strip-mining tiling for affine bands -*- C++
//-*-===//
//
// 目标:
//   读取 tile.sizes 属性, 对 perfect-nested 循环带逐层施加 strip-mining 分块.
//   外层生成块循环, 内层生成点循环, 点循环使用 scf.for 以便采用 SSA 上界.
//   尾块通过 affine.min 结合 scf.for 上界自动处理.
//
//
// 使用方式示例:
//   /usr/bin/mlir-opt-18 test2/input_tile.mlir -linalg-generalize-named-ops |
//   /usr/bin/mlir-opt-18 - -one-shot-bufferize="bufferize-function-boundaries"
//   | /usr/bin/mlir-opt-18 - -convert-linalg-to-affine-loops -canonicalize -cse
//   |
//   /usr/bin/mlir-opt-18 - \
//       -load-pass-plugin=./build/GpuTileILPSelection.so \
//       -pass-pipeline='builtin.module(func.func(gpu-apply-tiles))'
//
// 属性示例:
//   在最外层 affine.for 或 func 上附加:
//     tile.sizes = [8, 8, 32]
//
// 版本:
//   llvmorg-18.1.8
//
//===----------------------------------------------------------------------===//

#include <optional>
#include <string>
#include <vector>

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include "mlir/Support/LogicalResult.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "mlir/Tools/Plugins/PassPlugin.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {

/// 收集 perfect-nested 循环带.
/// 规则: 从 root 开始, 如果循环体内非终结操作恰好一个且为 affine.for,
/// 则继续向内收集.
static void collectPerfectLoopBand(affine::AffineForOp rootLoop,
                                   SmallVectorImpl<affine::AffineForOp> &band) {
  band.clear();
  if (!rootLoop)
    return;

  affine::AffineForOp currentLoop = rootLoop;
  while (true) {
    band.push_back(currentLoop);

    Block *bodyBlock = currentLoop.getBody();
    affine::AffineForOp innerLoop = nullptr;
    unsigned nonTerminatorCount = 0;

    for (Operation &operation : *bodyBlock) {
      if (isa<affine::AffineYieldOp>(operation))
        continue;
      ++nonTerminatorCount;
      if (nonTerminatorCount == 1)
        if (auto nestedFor = dyn_cast<affine::AffineForOp>(operation))
          innerLoop = nestedFor;
    }

    if (nonTerminatorCount == 1 && innerLoop)
      currentLoop = innerLoop;
    else
      break;
  }
}

/// 读取 tile.sizes 属性.
/// 优先从 band 最外层循环读取, 否则退回到当前函数读取.
/// 返回与 band 深度等长的整型向量. 若属性个数少于 band 深度, 则对缺失层使用 1.
/// 若属性不存在, 返回空向量表示不进行分块.
static SmallVector<int64_t, 8>
readTileSizesForBand(ArrayRef<affine::AffineForOp> band) {
  SmallVector<int64_t, 8> tileSizes;

  auto parseArray = [&](ArrayAttr arrayAttr) -> bool {
    tileSizes.clear();
    for (Attribute attr : arrayAttr) {
      if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
        tileSizes.push_back(intAttr.getInt());
      } else {
        return false;
      }
    }
    return true;
  };

  // 1) 优先: 最外层循环
  if (!band.empty()) {
    auto outer = band.front();
    if (auto arrayAttr = outer->getAttrOfType<ArrayAttr>("tile.sizes")) {
      if (parseArray(arrayAttr))
        goto normalize;
    }
  }

  // 2) 其次: 函数
  if (!band.empty()) {
    if (auto func = band.front()->getParentOfType<func::FuncOp>()) {
      if (auto arrayAttr = func->getAttrOfType<ArrayAttr>("tile.sizes")) {
        if (parseArray(arrayAttr))
          goto normalize;
      }
    }
  }

  // 未找到任何属性
  return {};

normalize:
  // 规范化到 band 深度长度. 缺失用 1 填充, 多余部分截断.
  if (tileSizes.size() < band.size())
    tileSizes.resize(band.size(), /*value=*/1);
  else if (tileSizes.size() > band.size())
    tileSizes.resize(band.size());
  // 将非正值归一为 1, 避免无意义的步长.
  for (int64_t &v : tileSizes)
    if (v <= 0)
      v = 1;
  return tileSizes;
}

/// 将一个 affine.for 做 strip-mining.
/// 生成:
///   affine.for %V = lb to ub step (tileSize * step) {
///     %minUb = affine.min ( %V + tileSize * step, ubExpr(..) )
///     scf.for %v = %V to %minUb step step {
///       ... clone old body, remapping old iv -> %v ...
///     }
///   }
/// 返回新创建的外层块循环, 同时删除原循环.
///
/// 注意:
///   1) 为了构造 SSA 上界, 使用 affine.min 生成 %minUb, 再用 scf.for.
///   2) 原循环体通过克隆进入新的点循环, 并 remap 原 induction var.
///   3) 本函数不做循环互换, 仅完成单层 strip-mining.
// 用纯 Affine 的实现替换原先的 affine.min + scf.for 版本.
// 关键点:
//   1) 点循环使用 affine.for, 保证索引仍然是 affine 的 dim/symbol.
//   2) 尾块用 affine.if 分两路: (V + T*s <= ub) 走“满块”上界, 否则走“裁剪到
//   ub”的上界. 3) 为了避免 dim 对齐的复杂度, 统一将上下界都用 symbol 形式
//   (s0)->(s0), 通过 operands 传 SSA 值.
//
static FailureOr<affine::AffineForOp>
stripMineSingleAffineFor(affine::AffineForOp oldFor, int64_t tileSize) {
  if (!oldFor || tileSize <= 1)
    return failure();

  OpBuilder builder(oldFor);
  Location loc = oldFor.getLoc();

  // 旧循环的上下界与步长
  AffineMap oldLowerMap = oldFor.getLowerBoundMap();
  AffineMap oldUpperMap = oldFor.getUpperBoundMap();
  SmallVector<Value, 8> oldLowerOperands(oldFor.getLowerBoundOperands().begin(),
                                         oldFor.getLowerBoundOperands().end());
  SmallVector<Value, 8> oldUpperOperands(oldFor.getUpperBoundOperands().begin(),
                                         oldFor.getUpperBoundOperands().end());
  int64_t oldStep = oldFor.getStep().getSExtValue();
  int64_t outerStep = oldStep * tileSize;

  // 外层块循环: step = tileSize * oldStep
  affine::AffineForOp outerFor = builder.create<affine::AffineForOp>(
      loc, oldLowerOperands, oldLowerMap, oldUpperOperands, oldUpperMap,
      outerStep);

  // 在外层体内构造:
  //   vPlus = V + tileSize*step
  //   ubVal = oldUpperMap(oldUpperOperands)
  //   if (vPlus <= ubVal) then full-tile else tail-tile
  OpBuilder bodyBuilder = OpBuilder::atBlockBegin(outerFor.getBody());
  Value outerInduction = outerFor.getInductionVar();

  // vPlus = V + outerStep
  AffineExpr d0 = bodyBuilder.getAffineDimExpr(0);
  AffineMap addCMap = AffineMap::get(
      /*dims=*/1, /*syms=*/0, d0 + bodyBuilder.getAffineConstantExpr(outerStep),
      bodyBuilder.getContext());
  Value vPlus = bodyBuilder.create<affine::AffineApplyOp>(
      loc, addCMap, ValueRange{outerInduction});

  // ubVal = apply old upper map
  Value ubVal = bodyBuilder.create<affine::AffineApplyOp>(
      loc, oldUpperMap, ValueRange(oldUpperOperands));

  // 构造 affine.if 条件: vPlus <= ubVal
  // 用 IntegerSet 表达: (d1 - d0) >= 0 也就是 ubVal - vPlus >= 0.
  {
    AffineExpr d0e = bodyBuilder.getAffineDimExpr(0); // vPlus
    AffineExpr d1e = bodyBuilder.getAffineDimExpr(1); // ubVal

    // constraints 只有一条: d1 - d0
    SmallVector<AffineExpr, 1> constraints;
    constraints.push_back(d1e - d0e);

    // eqFlags 同长度, false 表示这是不等式: (d1 - d0) >= 0
    SmallVector<bool, 1> eqFlags;
    eqFlags.push_back(false);

    IntegerSet condSet = IntegerSet::get(
        /*numDims=*/2, /*numSymbols=*/0,
        /*constraints=*/constraints,
        /*eqFlags=*/eqFlags);

    // operands 顺序与 dims 对齐: [vPlus, ubVal]
    auto ifOp = bodyBuilder.create<affine::AffineIfOp>(
        loc, condSet, ValueRange{vPlus, ubVal}, /*withElseRegion=*/true);

    // then: 满块, 上界 = V + tileStep, 全部用 dimension 表达式表示.
    // 维度 d0 对应外层 %V, 不需要也不能把 %V 当成 symbol 传入.
    {
      // 在 then block 的 terminator 之前插入
      Operation *thenTerm = ifOp.getThenBlock()->getTerminator();
      OpBuilder thenBuilder(thenTerm);

      AffineExpr d0i = thenBuilder.getAffineDimExpr(0);
      AffineMap lbDimId = AffineMap::get(/*dims=*/1, /*syms=*/0, d0i);
      AffineMap ubDimPlus =
          AffineMap::get(/*dims=*/1, /*syms=*/0,
                         d0i + thenBuilder.getAffineConstantExpr(outerStep));

      // 维度实参传入外层 induction 变量
      ValueRange dimOps{outerInduction};

      affine::AffineForOp innerFull = thenBuilder.create<affine::AffineForOp>(
          loc,
          /*lbOperands=*/dimOps, /*lbMap=*/lbDimId,
          /*ubOperands=*/dimOps, /*ubMap=*/ubDimPlus,
          /*step=*/oldStep);

      // 克隆旧体
      OpBuilder innerBuilder = OpBuilder::atBlockBegin(innerFull.getBody());
      IRMapping mapping;
      mapping.map(oldFor.getInductionVar(), innerFull.getInductionVar());
      for (Operation &op : *oldFor.getBody()) {
        if (isa<affine::AffineYieldOp>(op))
          continue;
        innerBuilder.clone(op, mapping);
      }
      // 注意: 不要在 then block 手动创建 affine.yield
    }

    // else: 尾块, 上界 = ubVal, 把 ubVal 作为 symbol 传入.
    // 注意: 仍保持 1 个维度 (对应外层 %V), 虽然表达式里未用到 d0.
    {
      Operation *elseTerm = ifOp.getElseBlock()->getTerminator();
      OpBuilder elseBuilder(elseTerm);

      AffineExpr d0i = elseBuilder.getAffineDimExpr(0);
      AffineMap lbDimId = AffineMap::get(/*dims=*/1, /*syms=*/0, d0i);

      // 上界仅使用 symbol, map 仍声明 1 个维度以匹配外层作用域
      AffineMap ubSymId = AffineMap::get(/*dims=*/1, /*syms=*/1,
                                         elseBuilder.getAffineSymbolExpr(0));

      // 维度与符号操作数: 先传外层维度, 再传 ubVal 这个 symbol
      SmallVector<Value, 2> ubOps{outerInduction, ubVal};
      ValueRange lbOps{outerInduction};

      affine::AffineForOp innerTail = elseBuilder.create<affine::AffineForOp>(
          loc,
          /*lbOperands=*/lbOps, /*lbMap=*/lbDimId,
          /*ubOperands=*/ubOps, /*ubMap=*/ubSymId,
          /*step=*/oldStep);

      OpBuilder innerBuilder = OpBuilder::atBlockBegin(innerTail.getBody());
      IRMapping mapping;
      mapping.map(oldFor.getInductionVar(), innerTail.getInductionVar());
      for (Operation &op : *oldFor.getBody()) {
        if (isa<affine::AffineYieldOp>(op))
          continue;
        innerBuilder.clone(op, mapping);
      }
      // 注意: 不要在 else block 手动创建 affine.yield
    }
  }

  // 用外层块循环替换旧循环, 并删除旧循环
  oldFor.replaceAllUsesWith(outerFor.getOperation());
  oldFor.erase();

  return outerFor;
}

/// 对一个 band 施加 strip-mining, 从最内层向外层逐层处理.
/// tileSizes 的长度等于 band 深度, 每一层对应一个 tile 大小.
/// 返回是否至少改变了一处循环.
static bool applyStripMiningToBand(ArrayRef<affine::AffineForOp> band,
                                   ArrayRef<int64_t> tileSizes) {
  if (band.empty())
    return false;

  bool changed = false;

  // 从最内层到最外层逐层处理, 便于在 IR 修改中保持稳定.
  for (int level = static_cast<int>(band.size()) - 1; level >= 0; --level) {
    affine::AffineForOp target = band[level];

    int64_t tileSize = tileSizes[level];
    if (tileSize <= 1)
      continue;

    FailureOr<affine::AffineForOp> result =
        stripMineSingleAffineFor(target, tileSize);
    if (succeeded(result)) {
      changed = true;
      // strip 后, 原 band[level] 已被替换为新的外层块循环.
      // 如果后续还要对更外层做处理, 不需要调整下标, 因为我们直接用原 band
      // 索引访问, 旧 handle 仍然有效于剩余层, 这是因为我们自内向外处理.
    }
  }

  return changed;
}

/// 主 Pass: 对函数内每个最外层 affine.for 作为根, 找到 band, 读取 tile.sizes,
/// 并应用 strip-mining.
struct ApplyTileSizesPass
    : public PassWrapper<ApplyTileSizesPass, OperationPass<func::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ApplyTileSizesPass)
  // 新增: 声明依赖方言. 构造 scf.for 与 arith.constant 必须加载这些方言.
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect>();
    registry.insert<scf::SCFDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<func::FuncDialect>(); // 保险起见, 虽然通常已加载
  }

  void runOnOperation() override {
    func::FuncOp function = getOperation();

    // 收集所有最外层 affine.for 作为 band 根
    SmallVector<affine::AffineForOp, 8> roots;
    function.walk([&](affine::AffineForOp forOp) {
      if (!forOp->getParentOfType<affine::AffineForOp>())
        roots.push_back(forOp);
    });

    if (roots.empty()) {
      llvm::errs() << "[apply-tiles] No affine.for found. "
                      "Please lower linalg to affine first.\n";
      return;
    }

    bool anyChange = false;

    for (auto root : roots) {
      SmallVector<affine::AffineForOp, 8> band;
      collectPerfectLoopBand(root, band);
      if (band.empty())
        continue;

      // 读取 tile.sizes 属性, 若缺失则跳过该 band
      SmallVector<int64_t, 8> tileSizes = readTileSizesForBand(band);
      if (tileSizes.empty())
        continue;

      // 打印 band 概览
      llvm::errs() << "[apply-tiles] Band depth = " << band.size()
                   << ", tile.sizes = [";
      for (unsigned i = 0; i < tileSizes.size(); ++i) {
        llvm::errs() << tileSizes[i];
        if (i + 1 < tileSizes.size())
          llvm::errs() << ", ";
      }
      llvm::errs() << "]\n";

      // 应用 strip-mining
      bool changed = applyStripMiningToBand(band, tileSizes);
      anyChange |= changed;
    }

    if (!anyChange) {
      // 非致命, 只是没有任何匹配的分块发生
      return;
    }
  }

  StringRef getArgument() const final { return "gpu-apply-tiles"; }
  StringRef getDescription() const final {
    return "Apply strip-mining tiling to affine perfect-nested bands using "
           "tile.sizes.";
  }
};

} // namespace

//=== 插件入口: 注册命名 Pass 与 Pipeline ===================================//

static std::unique_ptr<mlir::Pass> createApplyTileSizesPass() {
  return std::make_unique<ApplyTileSizesPass>();
}

extern "C" ::mlir::PassPluginLibraryInfo mlirGetPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION,
          /*pluginName=*/"ApplyTileSizes",
          /*pluginVersion=*/"v0.1-strip-mining-scf-inner",
          /*registerPassPipeline=*/
          []() {
            ::mlir::PassRegistration<ApplyTileSizesPass>();
            ::mlir::PassPipelineRegistration<>(
                "gpu-apply-tiles",
                "Strip-mining tiling for affine bands with scf inner loops.",
                [](::mlir::OpPassManager &pm) {
                  pm.addPass(createApplyTileSizesPass());
                });
          }};
}
