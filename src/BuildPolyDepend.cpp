//===- BuildPolyDepend.cpp - 基于 affine 循环带的依赖分析 -*- C++ -*-===//
//
// 目的: 在“依赖分析”这一步, 基于已识别出的 perfect-nested 循环带(band),
//       枚举带内的内存访问对, 使用 MLIR Affine 的分析工具判定是否存在真实依赖
//       (读后写、写后读、写后写), 并打印每个循环层的方向向量(direction vector)。
//
//
// 版本: llvmorg-18.1.8
//
// 使用方式:
//   # 先将 linalg 降至 affine
//   mlir-opt-18 input.mlir -linalg-generalize-named-ops |
//   mlir-opt-18 - -one-shot-bufferize="bufferize-function-boundaries" |
//   mlir-opt-18 - -convert-linalg-to-affine-loops -canonicalize -cse |
//   mlir-opt-18 - \
//     -load-pass-plugin=./build/GpuTileILPSelection.so \
//     -pass-pipeline='builtin.module(func.func(poly-depend-analyze))'
//
//===----------------------------------------------------------------------===//

#include <optional>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"

// 方言与算子
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

// Affine 依赖分析工具(提供 MemRefAccess 与依赖检查)
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"

// 插件入口
#include "mlir/Tools/Plugins/PassPlugin.h"

// LLVM 工具
#include <string>
#include <tuple>
#include <unordered_set>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {

/// 将方向分量转换为简洁字符串, 用作签名的一部分。
static std::string buildDirectionSignature(
    const llvm::SmallVector<mlir::affine::DependenceComponent, 2> &components) {
  std::string buffer;
  llvm::raw_string_ostream os(buffer);
  os << "<";
  for (unsigned index = 0; index < components.size(); ++index) {
    const auto &component = components[index];
    std::optional<int64_t> lower = component.lb;
    std::optional<int64_t> upper = component.ub;
    char symbol = '?';
    if (lower.has_value() && upper.has_value() && *lower == 0 && *upper == 0) {
      symbol = '=';
    } else if (upper.has_value() && *upper < 0) {
      symbol = '<';
    } else if (lower.has_value() && *lower > 0) {
      symbol = '>';
    } else if ((!lower.has_value() || *lower <= 0) &&
               (!upper.has_value() || *upper >= 0)) {
      symbol = '*';
    } else {
      symbol = '?';
    }
    os << symbol;
    if (index + 1 < components.size()) {
      os << ",";
    }
  }
  os << ">";
  return os.str();
}

/// 从根循环开始收集 perfect-nested 循环带。
static void collectPerfectLoopBand(affine::AffineForOp rootLoop,
                                   SmallVectorImpl<affine::AffineForOp> &band) {
  band.clear();
  if (!rootLoop) {
    return;
  }
  affine::AffineForOp currentLoop = rootLoop;
  while (true) {
    band.push_back(currentLoop);
    Block *bodyBlock = currentLoop.getBody();
    affine::AffineForOp innerLoop = nullptr;
    unsigned nonTerminatorCount = 0;
    for (Operation &op : *bodyBlock) {
      if (isa<affine::AffineYieldOp>(op)) {
        continue;
      }
      ++nonTerminatorCount;
      if (nonTerminatorCount == 1) {
        if (auto nestedFor = dyn_cast<affine::AffineForOp>(op)) {
          innerLoop = nestedFor;
        }
      }
    }
    if (nonTerminatorCount == 1 && innerLoop) {
      currentLoop = innerLoop;
    } else {
      break;
    }
  }
}

/// 内存访问描述。
struct MemoryAccessInfo {
  Operation *operation = nullptr;
  Value memory = nullptr;
  bool isWrite = false;
  affine::MemRefAccess asAccess;
  MemoryAccessInfo(Operation *op, bool write) : operation(op), isWrite(write), asAccess(op) {
    if (auto loadOp = dyn_cast<affine::AffineLoadOp>(op)) {
      memory = loadOp.getMemRef();
    } else if (auto storeOp = dyn_cast<affine::AffineStoreOp>(op)) {
      memory = storeOp.getMemRef();
    }
  }
};

/// 收集循环带体内的所有内存访问。
static void collectBandMemoryAccesses(ArrayRef<affine::AffineForOp> band,
                                      SmallVectorImpl<MemoryAccessInfo> &out) {
  out.clear();
  for (affine::AffineForOp loop : band) {
    Block *body = loop.getBody();
    body->walk([&](Operation *op) {
      if (auto loadOp = dyn_cast<affine::AffineLoadOp>(op)) {
        out.emplace_back(op, /*write=*/false);
      } else if (auto storeOp = dyn_cast<affine::AffineStoreOp>(op)) {
        out.emplace_back(op, /*write=*/true);
      }
    });
  }
}

/// 打印单个依赖分量的方向。
static void printDependenceComponentDirection(const mlir::affine::DependenceComponent &component,
                                              llvm::raw_ostream &os) {
  std::optional<int64_t> lower = component.lb;
  std::optional<int64_t> upper = component.ub;
  os << "[";
  if (lower.has_value()) {
    os << *lower;
  } else {
    os << "-inf";
  }
  os << ", ";
  if (upper.has_value()) {
    os << *upper;
  } else {
    os << "+inf";
  }
  os << "] ";
  if (lower.has_value() && upper.has_value() && *lower == 0 && *upper == 0) {
    os << "=";
    return;
  }
  if (upper.has_value() && *upper < 0) {
    os << "<";
    return;
  }
  if (lower.has_value() && *lower > 0) {
    os << ">";
    return;
  }
  if ((!lower.has_value() || *lower <= 0) && (!upper.has_value() || *upper >= 0)) {
    os << "*";
    return;
  }
  os << "?";
}

/// 执行依赖分析。
static void analyzeDependencesForBand(ArrayRef<affine::AffineForOp> loopBand) {
  if (loopBand.empty()) {
    return;
  }
  SmallVector<MemoryAccessInfo, 16> allAccesses;
  collectBandMemoryAccesses(loopBand, allAccesses);
  if (allAccesses.empty()) {
    llvm::errs() << "  [信息] 循环带中没有发现内存访问。\n";
    return;
  }
  llvm::DenseMap<Value, SmallVector<unsigned, 8>> accessesByMemory;
  for (unsigned index = 0; index < allAccesses.size(); ++index) {
    accessesByMemory[allAccesses[index].memory].push_back(index);
  }
  unsigned loopDepth = loopBand.size();
  struct KeyHash {
    std::size_t operator()(const std::tuple<void *, bool, bool, std::string> &k) const {
      auto h1 = std::hash<void *>()(std::get<0>(k));
      auto h2 = std::hash<bool>()(std::get<1>(k));
      auto h3 = std::hash<bool>()(std::get<2>(k));
      auto h4 = std::hash<std::string>()(std::get<3>(k));
      return (((h1 * 131 + h2) * 131 + h3) * 131) ^ h4;
    }
  };
  std::unordered_set<std::tuple<void *, bool, bool, std::string>, KeyHash> printed;
  for (auto &pair : accessesByMemory) {
    Value memrefValue = pair.first;
    auto &indices = pair.second;
    if (indices.size() <= 1) {
      continue;
    }
    for (unsigned a = 0; a < indices.size(); ++a) {
      for (unsigned b = a + 1; b < indices.size(); ++b) {
        Operation *operationA = allAccesses[indices[a]].operation;
        Operation *operationB = allAccesses[indices[b]].operation;
        bool aBeforeB = operationA->isBeforeInBlock(operationB);
        const MemoryAccessInfo &source =
            aBeforeB ? allAccesses[indices[a]] : allAccesses[indices[b]];
        const MemoryAccessInfo &destination =
            aBeforeB ? allAccesses[indices[b]] : allAccesses[indices[a]];
        if (!source.isWrite && !destination.isWrite) {
          continue;
        }
        llvm::SmallVector<mlir::affine::DependenceComponent, 2> components;
        mlir::affine::DependenceResult result = mlir::affine::checkMemrefAccessDependence(
            source.asAccess,
            destination.asAccess,
            loopDepth,
            nullptr,
            &components,
            false);
        bool hasDependence = (result.value != mlir::affine::DependenceResult::NoDependence);
        if (!hasDependence) {
          continue;
        }
        std::string signature = buildDirectionSignature(components);
        auto key = std::make_tuple(memrefValue.getAsOpaquePointer(),
                                   source.isWrite,
                                   destination.isWrite,
                                   signature);
        if (!printed.insert(std::move(key)).second) {
          continue;
        }
        const char *kind = "UnknownDependence";
        if (source.isWrite && !destination.isWrite) {
          kind = "WriteAfterRead? (source write, destination read)";
        }
        if (source.isWrite && destination.isWrite) {
          kind = "WriteAfterWrite";
        }
        if (!source.isWrite && destination.isWrite) {
          kind = "ReadAfterWrite? (source read, destination write)";
        }

        llvm::errs() << "  [Dependence Found] between "
                     << (source.isWrite ? "Store" : "Load") << " (source) and "
                     << (destination.isWrite ? "Store" : "Load")
                     << " (destination), kind = " << kind << "\n";

        llvm::errs() << "    Direction Vector per Loop Level: ";
        llvm::errs() << signature << "\n";

        bool hasStrictOrder = false;
        for (const auto &c : components) {
          std::optional<int64_t> lower = c.lb;
          std::optional<int64_t> upper = c.ub;
          if ((upper.has_value() && *upper < 0) || (lower.has_value() && *lower > 0)) {
            hasStrictOrder = true;
            break;
          }
        }
        if (!hasStrictOrder) {
          llvm::errs() << "    [提示] 没有严格 < 或 > 的层, 可能可并行。\n";
        } else {
          llvm::errs() << "    [提示] 存在严格顺序层, 可能限制重排和并行化。\n";
        }
      }
    }
  }
}

/// Pass 实现。
struct BuildPolyDependPass : public PassWrapper<BuildPolyDependPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BuildPolyDependPass)
  void runOnOperation() override {
    func::FuncOp function = getOperation();
    SmallVector<affine::AffineForOp, 8> roots;
    function.walk([&](affine::AffineForOp forOp) {
      if (!forOp->getParentOfType<affine::AffineForOp>()) {
        roots.push_back(forOp);
      }
    });
    if (roots.empty()) {
      llvm::errs() << "[依赖] 未发现 affine.for, 请先将 linalg 转 affine。\n";
      return;
    }
    int bandCounter = 0;
    for (auto root : roots) {
      SmallVector<affine::AffineForOp, 8> band;
      collectPerfectLoopBand(root, band);
      llvm::errs() << "\n[依赖] 循环带 #" << bandCounter++ << " (深度 "
                   << band.size() << ")\n";
      for (unsigned i = 0; i < band.size(); ++i) {
        auto forOp = band[i];
        llvm::errs() << "  L" << i << ": for %induction in ["
                     << forOp.getLowerBoundMap() << ", "
                     << forOp.getUpperBoundMap() << ") step "
                     << forOp.getStep() << "\n";
      }
      analyzeDependencesForBand(band);
    }
  }
  StringRef getArgument() const final { return "poly-depend-analyze"; }
  StringRef getDescription() const final {
    return "对每个 perfect-nested 循环带进行内存访问依赖分析, 并打印方向向量。";
  }
};

}  // namespace

static std::unique_ptr<mlir::Pass> createBuildPolyDependPass() {
  return std::make_unique<BuildPolyDependPass>();
}

extern "C" ::mlir::PassPluginLibraryInfo mlirGetPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION,
          "BuildPolyDepend",
          "v1.0",
          []() {
            ::mlir::PassRegistration<BuildPolyDependPass>();
            ::mlir::PassPipelineRegistration<>(
                "poly-depend-analyze",
                "分析每个 affine perfect-nested 循环带的内存访问依赖, 并打印方向向量。",
                [](::mlir::OpPassManager &pm) { pm.addPass(createBuildPolyDependPass()); });
          }};
}
