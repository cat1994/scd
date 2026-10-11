#ifndef HR_EDL_ACTION_TRANSFORMATION_H_
#define HR_EDL_ACTION_TRANSFORMATION_H_

#include <cassert>

#include "eigen/Eigen/Dense"
#include "hr_edl/math.h"
#include "hr_edl/types.h"

namespace hr_edl {

// 生成一个单位矩阵的对角线索引向量 [0, 1, 2, ..., d-1]
inline std::vector<size_t> SquareIdentity(size_t d) {
  std::vector<size_t> eye;
  eye.reserve(d);
  for (size_t i = 0; i < d; ++i) {
    eye.push_back(i);
  }
  return eye;
}

// 判断phi_block是否表示一个外部变换
// 外部变换：所有动作都映射到同一个动作 
inline bool RepresentsExternalTransformation(
    const std::vector<size_t>& phi_block) {
  const size_t ex_a = phi_block[0]; // 指向同一个动作索引
  for (size_t a = 1; a < phi_block.size(); ++a) {
    if (phi_block[a] != ex_a) {
      return false;
    }
  }
  return true;
}

// SwapActionTranformation 类用于处理动作的转换，
// phi_block_ 是一个动作映射向量，表示每个动作被映射到哪个动作 ai->aj, 
// e.g. 外部转换 ->a0 {a0,a0,a0,...}, 内部转换 a1->a0 {a0,a0,a2,...}
// is_external_ 标记该转换是否为外部变换
class SwapActionTranformation {
 public:
  // 生成所有可能的外部变换列表 <SwapActionTranformation>
  static std::vector<SwapActionTranformation> External(size_t num_actions) {
    std::vector<SwapActionTranformation> list;
    list.reserve(num_actions);
    for (size_t a1 = 0; a1 < num_actions; ++a1) {
      list.emplace_back(std::vector<size_t>(num_actions, a1));
    }
    return list;
  }
  // 生成所有可能的内部变换列表 <SwapActionTranformation>
  static std::vector<SwapActionTranformation> Internal(size_t num_actions) {
    std::vector<SwapActionTranformation> list;
    list.reserve(num_actions * num_actions - num_actions); // n*n-n 种内部转换类型
    std::vector<size_t> phi_block = SquareIdentity(num_actions); // 类似python range(d)
    for (size_t a1 = 0; a1 < num_actions; ++a1) {
      for (size_t a2 = 0; a2 < num_actions; ++a2) {
        if (a1 == a2) { // 同种动作不计入转换
          continue;
        }
        phi_block[a1] = a2; // 交换动作a1->a2
        list.emplace_back(phi_block); // 添加一个新的转换
        phi_block[a1] = a1; // 还原动作映射向量
      }
    }
    return list;
  }

 public:
  // 构造函数，接受一个右值引用 动作映射向量
  SwapActionTranformation(std::vector<size_t>&& phi_block)
      : phi_block_(std::move(phi_block)),
        is_external_(RepresentsExternalTransformation(phi_block_)) {}
  // 构造函数，接受一个常量引用 动作映射向量
  SwapActionTranformation(const std::vector<size_t>& phi_block)
      : phi_block_(phi_block),
        is_external_(RepresentsExternalTransformation(phi_block_)) {}
  virtual ~SwapActionTranformation() = default;

  // 纯策略，根据交换后的动作索引ai->aj生成一个新的动作策略向量{0,aj=1,others=0}
  std::vector<double> operator()(size_t action) const {
    std::vector<double> phi_policy(NumActions(), 0); // 初始化为0向量
    phi_policy[SwapTarget(action)] = 1.0; // 目标动作索引位置为1
    return phi_policy;
  }
  // 随机策略，根据phi_block生成转换后的策略
  std::vector<double> operator()(const std::vector<double>& policy) const {
    std::vector<double> phi_policy(policy.size(), 0);
    for (size_t action = 0; action < NumActions(); ++action) {
      phi_policy[SwapTarget(action)] += policy[action];
    }
    return phi_policy;
  }

  // 获取当前动作aidx交换后的动作索引
  size_t SwapTarget(size_t action) const { return phi_block_[action]; }
  size_t NumActions() const { return phi_block_.size(); }

  // 计算给定策略policy在该转换下策略的遗憾值
  double Regret(const CfValues& cfvs, const std::vector<double>& policy) const {
    double r = -cfvs(); // -期望值，操作符重载
    assert(policy.size() == cfvs.Size());
    assert(policy.size() == NumActions());
    for (size_t action = 0; action < policy.size(); ++action) { // 转后后策略的期望值-期望值
      r += cfvs[phi_block_[action]] * policy[action]; // 累加期望值：转换后动作的cfv值乘以该动作的概率
    }
    return r;
  }

 private:
  const std::vector<size_t> phi_block_; // 动作映射向量 ai->aj

 public:
  const bool is_external_; // 是否为外部变换
};

using Sat = SwapActionTranformation; // 简写

// WeightedActionTransformation 类用于处理加权的动作转换
// 维护一个加权和weight_sum_，一个加权的phi矩阵weighted
class WeightedActionTransformation {
 private:
 // 生成一个(n+1,n)维矩阵，其中(n,n)矩阵是负单位矩阵，（由于从0行开始计数）最后一行第n行全为1
  inline static Eigen::MatrixXd ProjectiveEye(size_t num_actions) {
    Eigen::MatrixXd a =
        -Eigen::MatrixXd::Identity(num_actions + 1, num_actions); // （n+1,n）维负单位矩阵，最后一行填充0
    a.row(num_actions) = Eigen::VectorXd::Constant(num_actions, 1.0); // 替换最后一行全为1
    return a;
  }
  // 生成一个n+1维约束向量 {0,0,...,1}
  inline static Eigen::VectorXd BVector(size_t num_actions) {
    Eigen::VectorXd b = Eigen::VectorXd::Zero(num_actions + 1); // 初始化为0向量
    b(num_actions) = 1.0; // 最后一个元素设置为1
    return b;
  }

 public:
 // 构造函数，接受动作数量，初始化加权矩阵和向量
  WeightedActionTransformation(size_t num_actions)
      : weight_sum_(0), // 初始化权重和为0
        weighted_phi_blocks_(Eigen::MatrixXd::Zero(num_actions, num_actions)), // 初始化加权phi矩阵为(n,n)维全0矩阵
        weighted_external_blocks_(Eigen::VectorXd::Zero(num_actions)), // 初始化加权外部变换向量为(n)维全0向量
        all_external_(true),
        projective_eye_(ProjectiveEye(num_actions)), // 生成投影单位矩阵
        b_(BVector(num_actions)) {} // 生成约束向量 {0,0,...,1}
  virtual ~WeightedActionTransformation() = default;

  // 添加一个转换phi，权重默认为1.0
  void Add(const Sat& phi, double weight = 1.0) {
    weight_sum_ += weight;
    if (phi.is_external_) {
      weighted_external_blocks_(phi.SwapTarget(0)) += weight; // 外部交换向量为全目标index值，只需更新对应动作的权重
    } else {
      all_external_ = false;
      for (size_t action = 0; action < NumActions(); ++action) {
        weighted_phi_blocks_(phi.SwapTarget(action), action) += weight; // 内部交换更新权重矩阵(aj,ai)位置权重值
      }
    }
  }

  // 将内部转换权重矩阵和外部转换权重向量相加，得到最终的转换权重矩阵
  // e.g. [[m11+v1, m12+v1,...], [m21+v2, m22+v2,...],...]
  // weighted_external_blocks_ 实际上可以转化为内部权重矩阵，每行都是{vi,...}，表示所有动作映射到动作ai的权重
  Eigen::MatrixXd ToMatrix() const {
    return weighted_phi_blocks_.colwise() + weighted_external_blocks_; 
  }

  // 计算转换矩阵的一个不动点，返回一个策略向量
  std::vector<double> FixedPoint() const {
    std::vector<double> pi_mem(NumActions(), 1.0 / NumActions()); // 初始化为n维均匀分布
    FixedPoint(pi_mem);
    return pi_mem;
  }

  // 计算转换矩阵的一个不动点策略，结果存储在pi_mem中， 权重对应了该phi的加权遗憾值w^{t+1} * (rho(phi))^+
  void FixedPoint(std::vector<double>& pi_mem) const {
    if (weight_sum_ > 0) {
      Eigen::Map<Eigen::VectorXd> pi(pi_mem.data(), NumActions()); // 将pi_mem映射为Eigen向量pi，pi_mem.data()返回一个指向数据的指针double*
      if (all_external_) { // 如果所有添加的变换都是外部变换
        pi = weighted_external_blocks_ / weight_sum_; // 直接归一化外部变换向量
      } else { // 否则需要求解线性方程组 Ax=b
        Eigen::MatrixXd A = projective_eye_; // 生成(n+1,n)维 负单位矩阵
        A.topRows(NumActions()) += ToMatrix() / weight_sum_; // 每一列加上 **归一化** 的转换矩阵
        pi = (A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b_)) // 求解线性方程组 Ax=b
                 .cwiseMax(0)
                 .cwiseMin(1.0); // 保证结果在[0,1]范围内，这个过程是什么？
      }
    } else {
      pi_mem.assign(NumActions(), 1.0 / NumActions()); // 如果权重和为0，返回均匀分布
    }
  }

  // 获取权重和
  double WeightSum() const { return weight_sum_; }
  // 重置加权矩阵和向量
  void Reset() {
    weight_sum_ = 0;
    weighted_phi_blocks_.setZero();
    weighted_external_blocks_.setZero();
  }

  // 获取动作数量
  size_t NumActions() const { return weighted_phi_blocks_.rows(); }

 private:
  double weight_sum_; // 权重和
  Eigen::MatrixXd weighted_phi_blocks_; // 内部转换的权重矩阵
  Eigen::VectorXd weighted_external_blocks_; // 外部变换的权重向量
  bool all_external_; // 是否所有添加的变换都是外部变换
  const Eigen::MatrixXd projective_eye_; // （n+1,n）负投影单位矩阵，最后一行全为1
  const Eigen::VectorXd b_; // (n+1)维约束向量 {0,0,...,1}
};

}  // namespace hr_edl
#endif  // HR_EDL_ACTION_TRANSFORMATION_H_
