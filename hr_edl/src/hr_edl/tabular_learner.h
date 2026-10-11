#ifndef HR_EDL_TABULAR_LEARNER_H_
#define HR_EDL_TABULAR_LEARNER_H_

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "hr_edl/action_transformation.h"

namespace hr_edl {

struct StageExposureEntry {
  int player = -1;
  std::string info_state;
  bool is_external = false;
  size_t g = 0;
  double w = 0.0;
};

// Statistics whose accumulation window is exactly one adaptive-epsilon stage.
// Solver regrets are deliberately not represented here: they have lifetime
// semantics and are normally retained when a stage is changed.
struct StageRegretMetrics {
  double max_transformed_full_regret = 0.0;
  double max_original_full_regret = 0.0;
  double max_conditional_regret = 0.0;
  double sum_positive_conditional_regret = 0.0;
  double w_min = std::numeric_limits<double>::infinity();
  double w_min_positive = std::numeric_limits<double>::infinity();
  double w_sum = 0.0;
  double w_max = 0.0;
  double w_at_max_csr = 0.0;
  size_t num_w = 0;
  size_t num_positive_w = 0;
  size_t num_zero_or_tiny_w = 0;
  std::vector<StageExposureEntry> exposure_entries;
  bool has_bottleneck = false;
  int bottleneck_player = -1;
  std::string bottleneck_info_state;
  bool bottleneck_is_external = false;
  size_t bottleneck_g = 0;

  void AddW(double w, double numerical_eps) {
    w_min = std::min(w_min, w);
    w_max = std::max(w_max, w);
    w_sum += w;
    ++num_w;
    if (w > numerical_eps) {
      w_min_positive = std::min(w_min_positive, w);
      ++num_positive_w;
    } else {
      ++num_zero_or_tiny_w;
    }
  }

  void Merge(const StageRegretMetrics& other) {
    max_transformed_full_regret =
        std::max(max_transformed_full_regret, other.max_transformed_full_regret);
    max_original_full_regret =
        std::max(max_original_full_regret, other.max_original_full_regret);
    if (other.has_bottleneck &&
        (!has_bottleneck || other.max_conditional_regret > max_conditional_regret)) {
      max_conditional_regret = other.max_conditional_regret;
      w_at_max_csr = other.w_at_max_csr;
      has_bottleneck = true;
      bottleneck_player = other.bottleneck_player;
      bottleneck_info_state = other.bottleneck_info_state;
      bottleneck_is_external = other.bottleneck_is_external;
      bottleneck_g = other.bottleneck_g;
    }
    sum_positive_conditional_regret += other.sum_positive_conditional_regret;
    w_min = std::min(w_min, other.w_min);
    w_min_positive = std::min(w_min_positive, other.w_min_positive);
    w_sum += other.w_sum;
    w_max = std::max(w_max, other.w_max);
    num_w += other.num_w;
    num_positive_w += other.num_positive_w;
    num_zero_or_tiny_w += other.num_zero_or_tiny_w;
    exposure_entries.insert(exposure_entries.end(), other.exposure_entries.begin(),
                            other.exposure_entries.end());
  }

  void Finalize() {
    if (!std::isfinite(w_min)) w_min = 0.0;
    if (!std::isfinite(w_min_positive)) w_min_positive = 0.0;
  }

  double WMean() const { return num_w == 0 ? 0.0 : w_sum / num_w; }
  bool HasValidConditionalExposure() const { return num_zero_or_tiny_w == 0; }
};

namespace _tl {
// 接口，定义了一个立即响应的策略
// 该接口用于获取当前状态下的动作概率分布
class ImmediatePolicy {
 public:
  virtual ~ImmediatePolicy() = default;
  virtual std::vector<double> Response() const = 0;  // 返回策略向量
  virtual void Response(double* out) const = 0;      // 将策略存入out指针
  virtual size_t NumActions() const = 0;
};

// CachedImmediatePolicy 类，实现了 ImmediatePolicy
// 接口，缓存了一个信息状态is的固定策略
class CachedImmediatePolicy : public ImmediatePolicy {
 public:
  // 构造函数，接受动作数量或一个策略向量
  CachedImmediatePolicy(size_t num_actions) : policy_(num_actions, 1.0 / num_actions) {}
  CachedImmediatePolicy(std::vector<double>&& policy) : policy_(std::move(policy)) {}
  virtual ~CachedImmediatePolicy() = default;

  // 返回缓存的策略向量
  std::vector<double> Response() const override final { return policy_; }
  // 将策略存入out指针，out指向的内存必须足够大
  void Response(double* out) const override final {
    assert(out);
    std::memcpy(out, policy_.data(), policy_.size() * sizeof(*out));
  }
  // 返回动作数量
  size_t NumActions() const override final { return policy_.size(); }

 protected:
  std::vector<double> policy_;
};

// RegretTransformation 函数类型，接受
// 前一轮遗憾值、当前遗憾值和轮数，返回更新后的遗憾值
using RegretTransformation =
    std::function<double(double prev_regret, double regret, size_t round_number)>;
// LinkFn 函数类型，接受一个遗憾值向量和一个ReLU函数
// using EnumerationConsumer = std::function<void(int, T)>;
using LinkFn = std::function<void(const std::vector<double>&, const EnumerationConsumer<double>&)>;

// ReachProbLists 结构体，存储当前节点的概率列表
struct ReachProbLists {
  const std::vector<double> prev_;               // w^t. t轮次的memory state 概率
  const std::vector<double> next_;               // w^{t+1} t+1轮次的memory state 概率
  const std::vector<size_t> remaining_queries_;  // 当前节点剩余的查询数量列表
  size_t Size() const { return prev_.size(); }
};

// 单个is的即时决策信息，提供了策略更新的方法Update()
// 该类继承了CachedImmediatePolicy，包含了当前is的策略policy_、内外部变换及其遗憾值，策略更新等信息
class ImmediateDecisionInfo : public CachedImmediatePolicy {
 public:
  // 构造函数，接受动作数量、外部前驱概率数量和内部前驱概率数量
  // 每个节点有2种变换形式，内部和外部，对应了内部和外部前驱概率，例如causal
  ImmediateDecisionInfo(size_t num_actions, size_t ex_num_pred_reach_probs,
                        size_t in_num_pred_reach_probs)
      : CachedImmediatePolicy(num_actions),       // 调用基类构造函数，初始化策略为均匀分布
        round_number_(1),                         // 轮数初始化为1
        ex_phi_list_(ex_num_pred_reach_probs > 0  // 如果有外部前驱概率，则生成所有外部变换
                         ? SwapActionTranformation::External(num_actions)  // 生成所有外部变换
                         : std::vector<SwapActionTranformation>()),        // 否则为空
        in_phi_list_(in_num_pred_reach_probs > 0  // 如果有内部前驱概率，则生成所有内部变换
                         ? SwapActionTranformation::Internal(num_actions)  // 生成所有内部变换
                         : std::vector<SwapActionTranformation>()),        // 否则为空
        ex_regrets_(ex_phi_list_.size() * ex_num_pred_reach_probs,
                    0),  // 平铺方式初始化外部变换遗憾值为0向量,转换数量*memory
                         // state数量
        in_regrets_(in_phi_list_.size() * in_num_pred_reach_probs, 0),
        org_ex_regrets_(ex_phi_list_.size() * ex_num_pred_reach_probs,
                        0),  // 平铺方式初始化外部变换原始遗憾值为0向量
        org_in_regrets_(in_phi_list_.size() * in_num_pred_reach_probs,
                        0),  // 平铺方式初始化内部变换原始遗憾值为0向量
        stage_ex_regrets_(ex_phi_list_.size() * ex_num_pred_reach_probs, 0),
        stage_in_regrets_(in_phi_list_.size() * in_num_pred_reach_probs, 0),
        stage_org_ex_regrets_(ex_phi_list_.size() * ex_num_pred_reach_probs, 0),
        stage_org_in_regrets_(in_phi_list_.size() * in_num_pred_reach_probs, 0),
        // reached_ex_regrets_(ex_phi_list_.size() * ex_num_pred_reach_probs,
        // 0),  // 平铺方式初始化外部变换到达遗憾值为0向量
        // reached_in_regrets_(in_phi_list_.size() * in_num_pred_reach_probs,
        // 0),  // 平铺方式初始化内部变换到达遗憾值为0向量
        reached_probs_sum_ex_(ex_num_pred_reach_probs,
                              0.0),  // 平铺方式初始化外部变换到达概率和为0向量
        reached_probs_sum_in_(in_num_pred_reach_probs, 0.0),
        stage_reached_probs_sum_ex_(ex_num_pred_reach_probs, 0.0),
        stage_reached_probs_sum_in_(in_num_pred_reach_probs, 0.0),
        phi_sum_(num_actions)
  // cumulative_policy_(num_actions, 0.0)
  {}  // 初始化累积策略为0向量
  virtual ~ImmediateDecisionInfo() = default;

  // 更新决策信息，计算当前策略的遗憾值，并更新phi_sum_
  // 使用ex_pred_reach_prob_lists和in_pred_reach_prob_lists来存储外部和内部的前驱概率列表
  // 使用update_target来更新遗憾值，f是一个链接函数，用于处理遗憾值
  void Update(const CfValues& cfvs, const RegretTransformation& update_target, const LinkFn& f,
              const ReachProbLists& ex_pred_reach_prob_lists,
              const ReachProbLists& in_pred_reach_prob_lists, double epsilon = 0.0,
              double iwrp = 0.0) {
    if (!std::isfinite(epsilon) || epsilon < 0.0) {
      throw std::invalid_argument("epsilon must be finite and non-negative");
    }
    // Update both lifetime diagnostics and exact, non-truncated stage-local
    // original-space accumulators before transforming the values.
    if (is_original_) {
      if (ex_pred_reach_prob_lists.Size() > 0) {
        UpdateUnperturbedPhiRegrets(org_ex_regrets_, reached_probs_sum_ex_,
                                    stage_org_ex_regrets_, stage_reached_probs_sum_ex_,
                                    ex_phi_list_, cfvs, update_target,
                                    ex_pred_reach_prob_lists, iwrp);
      }
      if (in_pred_reach_prob_lists.Size() > 0) {
        UpdateUnperturbedPhiRegrets(org_in_regrets_, reached_probs_sum_in_,
                                    stage_org_in_regrets_, stage_reached_probs_sum_in_,
                                    in_phi_list_, cfvs, update_target,
                                    in_pred_reach_prob_lists, iwrp);
      }
    }
    // 计算perturbed cfvs 和 转换回原始策略空间
    if (epsilon > 0.0) {
      size_t num_actions = policy_.size();
      if (num_actions * epsilon >= 1.0) {
        throw std::invalid_argument(
            "epsilon must satisfy epsilon < 1 / num_actions at every information set");
      }
      // 计算原始博弈中的phi_regrets
      CfValues perturbed_cfvs = cfvs;  // 复制cfvs
      double p_value = 0;
      for (size_t aidx = 0; aidx < num_actions; ++aidx) {
        p_value += cfvs[aidx] * epsilon;
      }
      for (size_t aidx = 0; aidx < num_actions; ++aidx) {
        perturbed_cfvs[aidx] = (1 - num_actions * epsilon) * cfvs[aidx] + p_value;
      }
      // 转换策略 到 转换博弈中
      for (size_t a = 0; a < num_actions; ++a) {
        policy_[a] = (policy_[a] - epsilon) / (1.0 - num_actions * epsilon);
      }
      is_original_ = false;  // 标记为扰动空间

      Update(perturbed_cfvs, update_target, f, ex_pred_reach_prob_lists, in_pred_reach_prob_lists,
             0.0, iwrp);  // 避免递归调用时重复扰动
      // 将转换博弈中的策略映射回原始策略空间
      // 计算扰动策略项x_epsilon
      // double x_epsilon = 0;
      double x_epsilon = epsilon;  // 因为 sum_{a} pi(a) = 1
      // for (size_t a = 0; a < num_actions; ++a) {
      //   // 实质 x_epsilon = epsilon * sum_{a} pi(a) = epsilon * 1 = epsilon,
      //   若更一般的epsilon是一个向量，区分epsilon[a] x_epsilon += policy_[a] *
      //   epsilon;
      // }
      // 更新策略为 (1 - num_actions*epsilon)*pi(a) + x_epsilon
      for (size_t a = 0; a < num_actions; ++a) {
        policy_[a] = (1.0 - num_actions * epsilon) * policy_[a] + x_epsilon;
      }
      is_original_ = true;  // 标记为原始空间
      return;
    }

    if (ex_pred_reach_prob_lists.Size() > 0) {  // 更新外部变换的遗憾值数组y
      UpdatePhiSetRegrets(ex_regrets_, stage_ex_regrets_, ex_phi_list_, cfvs,
                          update_target, f, ex_pred_reach_prob_lists);
    }
    if (in_pred_reach_prob_lists.Size() > 0) {  // 更新内部变换的遗憾值矩阵
      UpdatePhiSetRegrets(in_regrets_, stage_in_regrets_, in_phi_list_, cfvs,
                          update_target, f, in_pred_reach_prob_lists);
    }
    phi_sum_.FixedPoint(policy_);  // 计算phi_sum_的固定点策略，存储在policy_

    phi_sum_.Reset();  // 将所有phi的权重和清零，将外部phi，内部phi的加权数组/矩阵清零
    ++round_number_;
  }

  // 计算当前is的最大phi遗憾值
  // std::tuple<double, double, double, double> PhiRegret_(const CfValues& cfvs,
  //                                               const ReachProbLists& ex_pred_reach_prob_lists,
  //                                               const ReachProbLists& in_pred_reach_prob_lists,
  //                                               double iwrp) const {
  //   double max_full_phi_regret = 0.0;
  //   double max_phi_regret = 0.0;
  //   double sum_phi_regret = 0.0;
  //   // auto policy = policy_; // 获取当前策略
  //   // 计算外部变换的最大遗憾值,对应了blind变换,只需要计算最大的外部转换遗憾值
  //   if (ex_pred_reach_prob_lists.Size() > 0) {
  //     double max_ex_regret = 0.0;
  //     for (size_t i = 0; i < ex_phi_list_.size(); ++i) {
  //       const double regret =
  //           ex_phi_list_[i].Regret(cfvs, policy_);  // 计算该变换的遗憾值cfv: rho^{t}
  //       sum_phi_regret += std::max(0.0, regret);    // 累加所有外部变换的遗憾值
  //       if (max_ex_regret < regret) {
  //         max_ex_regret = regret;
  //       }
  //     }
  //     max_phi_regret = max_ex_regret;
  //     // std::cout << "Max External Regret: " << max_ex_regret << " iwrp: " <<
  //     // iwrp << std::endl; // debug 输出
  //     for (size_t j = 0; j < ex_pred_reach_prob_lists.Size(); ++j) {
  //       double r =
  //           ex_pred_reach_prob_lists.prev_[j] * max_ex_regret;  // 计算加权遗憾值 w^{t}*rho^{t}
  //       if (r > max_full_phi_regret) {
  //         max_full_phi_regret = r;
  //       }
  //     }
  //   }

  //   // 计算内部变换的最大遗憾值, 对应了informed变换,需要计算所有内部转换的遗憾值
  //   if (in_pred_reach_prob_lists.Size() > 0) {
  //     size_t idx = 0;
  //     size_t num_action = policy_.size();

  //     // double max_in_regret = 0.0;
  //     for (size_t i = 0; i < num_action; ++i) {  // 内部转换i->?
  //       double max_in_regret = 0.0;
  //       for (size_t j = 0; j < num_action - 1; ++j) {
  //         // if (i==j) continue;
  //         const size_t phi_idx = i * (num_action - 1) + j;
  //         const double regret =
  //             in_phi_list_[phi_idx].Regret(cfvs, policy_);  // 计算该内部变换的遗憾值
  //         sum_phi_regret += std::max(0.0, regret);          // 累加所有内部变换的遗憾值
  //         if (max_in_regret < regret) {
  //           max_in_regret = regret;
  //         }
  //       }
  //       if (max_in_regret > max_phi_regret) {
  //         max_phi_regret = max_in_regret;
  //       }
  //       for (size_t j = 0; j < in_pred_reach_prob_lists.Size(); ++j) {
  //         double r =
  //             in_pred_reach_prob_lists.prev_[j] * max_in_regret;  // 计算加权遗憾值 w^{t}*rho^{t}
  //         if (r > max_full_phi_regret) {
  //           max_full_phi_regret = r;
  //         }
  //       }
  //     }
  //   }

  //   return {max_full_phi_regret, max_phi_regret / iwrp,
  //           sum_phi_regret / iwrp};  // 返回最大遗憾值，除以iwrp 贝叶斯 r=r/sum_{h\in
  //                                    // I}(q_{-i}(h))  //TODO 修改为 （max,max/iwrp）
  // }

  // 获取phi_regrets
  // max_full_phi_regret_fus: 从 ex_regrets_ 和 in_regrets_ 中获取最大值，表示用于更新策略的最大遗憾值
  // max_full_phi_regret: 从 ex_regrets_ 和 in_regrets_ 中获取最大值，表示当前策略下的最大遗憾值
  // max_phi_regret: 从 reached_ex_regrets_ 和
  // reached_in_regrets_中获取最大值，表示在已经到达该节点的情况下的最大遗憾值
  // sum_phi_regret: 从reached_ex_regrets_ 和
  // reached_in_regrets_中获取总和，表示在已经到达该节点的情况下的总遗憾值
  std::tuple<double, double, double, double> PhiRegrets() const {
    double max_full_phi_regret_fus = 0.0;  // for update strategy osr(mu)
    double max_full_phi_regret = 0.0; // for current strategy osr(v)
    double max_phi_regret = 0.0; // for reached nodes csr(v)
    double sum_phi_regret = 0.0; //
    for (size_t i = 0; i < ex_regrets_.size(); ++i) {
      // 计算外部变换的遗憾值
      const double rs = ex_regrets_[i];
      const double r = org_ex_regrets_[i];
      const double w = reached_probs_sum_ex_[i % reached_probs_sum_ex_.size()];
      const double rr = w > 0.0 ? org_ex_regrets_[i] / w : 0.0;
      if (rs > max_full_phi_regret_fus) {
        max_full_phi_regret_fus = rs;
      }
      if (r > max_full_phi_regret) {
        max_full_phi_regret = r;
      }
      if (rr > max_phi_regret) {
        max_phi_regret = rr;
      }
      sum_phi_regret += std::max(0.0, rr);
    }
    // 计算内部变换的遗憾值
    for (size_t i = 0; i < in_regrets_.size(); ++i) {
      const double rs = in_regrets_[i];
      const double r = org_in_regrets_[i];
      const double w = reached_probs_sum_in_[i % reached_probs_sum_in_.size()];
      const double rr = w > 0.0 ? org_in_regrets_[i] / w : 0.0;
      if (rs > max_full_phi_regret_fus) {
        max_full_phi_regret_fus = rs;
      }
      if (r > max_full_phi_regret) {
        max_full_phi_regret = r;
      }
      if (rr > max_phi_regret) {
        max_phi_regret = rr;
      }
      sum_phi_regret += std::max(0.0, rr);
    }
    return {max_full_phi_regret_fus, max_full_phi_regret, max_phi_regret, sum_phi_regret};
  }

  StageRegretMetrics StageMetrics(double w_numerical_eps) const {
    StageRegretMetrics metrics;
    for (double regret : stage_ex_regrets_) {
      metrics.max_transformed_full_regret =
          std::max(metrics.max_transformed_full_regret, regret);
    }
    for (double regret : stage_in_regrets_) {
      metrics.max_transformed_full_regret =
          std::max(metrics.max_transformed_full_regret, regret);
    }

    const auto add_original_set =
        [&metrics, w_numerical_eps](const std::vector<double>& regrets,
                                    const std::vector<double>& reached_probs,
                                    bool is_external) {
          for (size_t g = 0; g < reached_probs.size(); ++g) {
            metrics.AddW(reached_probs[g], w_numerical_eps);
            metrics.exposure_entries.push_back(
                {-1, "", is_external, g, reached_probs[g]});
          }
          if (reached_probs.empty()) return;
          for (size_t i = 0; i < regrets.size(); ++i) {
            const double regret = regrets[i];
            metrics.max_original_full_regret =
                std::max(metrics.max_original_full_regret, regret);
            const size_t g = i % reached_probs.size();
            const double w = reached_probs[g];
            if (w <= w_numerical_eps) continue;
            const double conditional_regret = regret / w;
            if (conditional_regret > metrics.max_conditional_regret) {
              metrics.max_conditional_regret = conditional_regret;
              metrics.w_at_max_csr = w;
              metrics.has_bottleneck = true;
              metrics.bottleneck_is_external = is_external;
              metrics.bottleneck_g = g;
            } else if (!metrics.has_bottleneck) {
              metrics.w_at_max_csr = w;
              metrics.has_bottleneck = true;
              metrics.bottleneck_is_external = is_external;
              metrics.bottleneck_g = g;
            }
            metrics.sum_positive_conditional_regret +=
                std::max(0.0, conditional_regret);
          }
        };
    add_original_set(stage_org_ex_regrets_, stage_reached_probs_sum_ex_, true);
    add_original_set(stage_org_in_regrets_, stage_reached_probs_sum_in_, false);
    metrics.Finalize();
    return metrics;
  }

  void ResetStageEvaluation() {
    std::fill(stage_ex_regrets_.begin(), stage_ex_regrets_.end(), 0.0);
    std::fill(stage_in_regrets_.begin(), stage_in_regrets_.end(), 0.0);
    std::fill(stage_org_ex_regrets_.begin(), stage_org_ex_regrets_.end(), 0.0);
    std::fill(stage_org_in_regrets_.begin(), stage_org_in_regrets_.end(), 0.0);
    std::fill(stage_reached_probs_sum_ex_.begin(), stage_reached_probs_sum_ex_.end(), 0.0);
    std::fill(stage_reached_probs_sum_in_.begin(), stage_reached_probs_sum_in_.end(), 0.0);
  }

  void ResetSolverRegrets() {
    std::fill(ex_regrets_.begin(), ex_regrets_.end(), 0.0);
    std::fill(in_regrets_.begin(), in_regrets_.end(), 0.0);
    phi_sum_.Reset();
    round_number_ = 1;
  }

  void SetCfrPlus(bool enabled) { is_cfr_plus_ = enabled; }

  // Untruncated original local regrets since the last stage reset. Fixed
  // tournaments never reset these accumulators, so this covers the whole run.
  double StageOriginalLocalRegret() const {
    double maximum = 0.0;
    for (double r : stage_org_ex_regrets_) maximum = std::max(maximum, r);
    for (double r : stage_org_in_regrets_) maximum = std::max(maximum, r);
    return maximum;
  }

  double SolverRegretL1() const {
    double total = 0.0;
    for (double r : ex_regrets_) total += std::abs(r);
    for (double r : in_regrets_) total += std::abs(r);
    return total;
  }

  double StageOriginalRegretL1() const {
    double total = 0.0;
    for (double r : stage_org_ex_regrets_) total += std::abs(r);
    for (double r : stage_org_in_regrets_) total += std::abs(r);
    return total;
  }

  double StageExposureSum() const {
    double total = 0.0;
    for (double w : stage_reached_probs_sum_ex_) total += w;
    for (double w : stage_reached_probs_sum_in_) total += w;
    return total;
  }

 private:
  // 更新phi集对应的即时遗憾值，将其加权到累积遗憾值上
  void UpdatePhiSetRegrets(std::vector<double>& regrets,
                           std::vector<double>& stage_regrets,
                           const std::vector<SwapActionTranformation>& phi_list,
                           const CfValues& cfvs, const RegretTransformation& update_target,
                           const LinkFn& f, const ReachProbLists& pred_reach_prob_lists) {
    const size_t num_reach_probs = pred_reach_prob_lists.prev_.size();
    assert(num_reach_probs ==
           pred_reach_prob_lists.next_.size());  // 检查w^t,w^{t+1}概率列表大小应相等
    size_t idx = 0;
    // regrets[idx]:  [phi_1 with w_1, phi_1 with w_2,...,phi_1 with w_m, phi_2
    // with w_1,...,phi_2 with w_m,...]
    for (size_t i = 0; i < phi_list.size(); ++i) {
      const double regret = phi_list[i].Regret(cfvs, policy_);  // 计算该变换的遗憾值cfv: rho^{t}
      for (size_t j = 0; j < num_reach_probs; ++j) {
        assert(idx < regrets.size());
        // 计算加权遗憾值 w^{t}*rho^{t}，并累加到累积遗憾值regrets,这个值保留
        // ltbr:
        // update_target函数这里直接返回第二个参数：regret，但可以被扩展，例如实现discounted
        // regret
        const double instantaneous_regret = pred_reach_prob_lists.prev_[j] * regret;
        regrets[idx] += update_target(regrets[idx], instantaneous_regret, round_number_);
        stage_regrets[idx] += instantaneous_regret;
        // NOTE: CFR+, relu 更稳定. 在SCD中会导致k值影响消失
        if (is_cfr_plus_ && regrets[idx] < 0.0) {
          regrets[idx] = 0.0;
        }
        ++idx;
      }
    }
    // 当前处理的前驱概率w索引
    size_t reach_prob_idx = 0;
    // 当前处理的phi索引
    size_t phi_idx = 0;
    // 累积每个phi的加权遗憾值 w^{t+1} * (rho^{phi})^+
    double sum = 0;
    // 更新外部phi或内部phi的weighted_external_blocks_ or weighted_phi_blocks_
    // 通过计算每个phi的链接值，y^{t+1}_{phi} = sum_{i} w^{t+1}_{i} *
    // (rho(phi))^+ f是链接函数Relu，执行每个phi的链接值计算
    f(regrets, [this, num_reach_probs, &reach_prob_idx, p_next = pred_reach_prob_lists.next_.data(),
                &sum, &phi_idx, &phi_list](size_t regret_idx, double link_output) {
      // y^{t+1}_{phi} += w^{t+1}_{phi} * (rho(phi))^+
      sum += p_next[reach_prob_idx] * link_output;
      ++reach_prob_idx;
      if (reach_prob_idx == num_reach_probs) {  // 遍历完一个phi的所有前驱概率
        assert(phi_idx < phi_list.size());
        // 将外部或者内部phi 及其权重和（state memory
        // probability）加入phi_sum_中
        phi_sum_.Add(phi_list[phi_idx],
                     sum);  // 将变换和加权和加入phi_sum_，更新weighted_external_blocks_
                            // or weighted_phi_blocks_
        ++phi_idx;
        sum = 0;
        reach_prob_idx = 0;
      }
    });
  }

  // 更新未扰动的phi遗憾值，将其加权到累积遗憾值上，只用于评估，不用于更新策略
  void UpdateUnperturbedPhiRegrets(std::vector<double>& regrets,
                                   std::vector<double>& reached_probs_sum,
                                   std::vector<double>& stage_regrets,
                                   std::vector<double>& stage_reached_probs_sum,
                                   const std::vector<SwapActionTranformation>& phi_list,
                                   const CfValues& cfvs, const RegretTransformation& update_target,
                                   const ReachProbLists& pred_reach_prob_lists,
                                   const double iwrp = 0.0) {
    const size_t num_reach_probs = pred_reach_prob_lists.prev_.size();
    assert(num_reach_probs ==
           pred_reach_prob_lists.next_.size());  // 检查w^t,w^{t+1}概率列表大小应相等
    size_t idx = 0;
    // regrets[idx]:  [phi_1 with w_1, phi_1 with w_2,...,phi_1 with w_m, phi_2
    // with w_1,...,phi_2 with w_m,...]
    // 更新累积到达概率
    for (size_t j = 0; j < num_reach_probs; ++j) {
      const double exposure = pred_reach_prob_lists.prev_[j] * iwrp;
      reached_probs_sum[j] += exposure;
      stage_reached_probs_sum[j] += exposure;
    }
    for (size_t i = 0; i < phi_list.size(); ++i) {
      const double regret = phi_list[i].Regret(cfvs, policy_);  // 计算该变换的遗憾值cfv: rho^{t}
      for (size_t j = 0; j < num_reach_probs; ++j) {
        assert(idx < regrets.size());
        // 计算加权遗憾值 w^{t}*rho^{t}，并累加到累积遗憾值regrets,这个值保留
        // ltbr:
        // update_target函数这里直接返回第二个参数：regret，但可以被扩展，例如实现discounted
        // regret
        const double instantaneous_regret = pred_reach_prob_lists.prev_[j] * regret;
        regrets[idx] += update_target(regrets[idx], instantaneous_regret, round_number_);
        // Stage evaluation is the exact sum of instantaneous original-space
        // regrets; it intentionally does not inherit CFR+ truncation.
        stage_regrets[idx] += instantaneous_regret;
        // 假设到达该节点的遗憾值reached_regrets,这个值保留
        // reached_regrets[idx] += update_target(
        // reached_regrets[idx], pred_reach_prob_lists.prev_[j] * regret / iwrp, round_number_);
        // NOTE: CFR+, relu 更稳定. 在SCD中会导致k值影响消失
        if (is_cfr_plus_) {
          if (regrets[idx] < 0.0) {
            regrets[idx] = 0.0;
          }
          // if (reached_regrets[idx] < 0.0) {
          //   reached_regrets[idx] = 0.0;
          // }
        }
        ++idx;
      }
    }
  }

 private:
  size_t round_number_;                                     // 当前轮数
  const std::vector<SwapActionTranformation> ex_phi_list_;  // 外部变换列表
  const std::vector<SwapActionTranformation> in_phi_list_;  // 内部变换列表
  std::vector<double> ex_regrets_;                          // 外部变换的累积遗憾值
  std::vector<double> in_regrets_;                          // 内部变换的累积遗憾值
  // 原始空间的累积遗憾值
  std::vector<double> org_ex_regrets_;  // 原始空间的外部累积遗憾值
  std::vector<double> org_in_regrets_;  // 原始空间的内部累积遗憾值
  std::vector<double> stage_ex_regrets_;
  std::vector<double> stage_in_regrets_;
  std::vector<double> stage_org_ex_regrets_;
  std::vector<double> stage_org_in_regrets_;
  // 累积遗憾值，假设已经到达该节点
  // std::vector<double> reached_ex_regrets_;
  // std::vector<double> reached_in_regrets_;
  std::vector<double> reached_probs_sum_ex_;  // 到达该节点的外部累积到达概率
  std::vector<double> reached_probs_sum_in_;  // 到达该节点的内部累积到达概率
  std::vector<double> stage_reached_probs_sum_ex_;
  std::vector<double> stage_reached_probs_sum_in_;
  WeightedActionTransformation phi_sum_;      // 加权动作变换类，用于更新策略
  // 累积策略, 用于计算平均策略
  //  std::vector<double> cumulative_policy_;
  // std::vector<double> perturbed_policy_;  // 扰动的当前策略
  bool is_original_ = true;  // 是否为原始空间的策略
  bool is_cfr_plus_ = true;  // Whether cumulative solver/lifetime regrets use CFR+.
};
}  // namespace _tl

// 定义各类deviation的前驱和后继节点的到达概率计算方法memory state probs: w^t,
// w^{t+1}
struct NoExternal {
  std::vector<double> ExternalPredecessorReachProbs(
      const std::vector<double>& pred_reach_probs,
      const std::vector<size_t>& pred_remain_queries) const {
    return {};
  }
};
struct AllExternal {
  std::vector<double> ExternalPredecessorReachProbs(
      const std::vector<double>& pred_reach_probs,
      const std::vector<size_t>& pred_remain_queries) const {
    return pred_reach_probs;
  }
};
struct NoInternal {
  std::vector<double> InternalPredecessorReachProbs(
      const std::vector<double>& pred_reach_probs,
      const std::vector<size_t>& pred_remain_queries) const {
    return {};
  }
};
struct AllInternal {
  std::vector<double> InternalPredecessorReachProbs(
      const std::vector<double>& pred_reach_probs,
      const std::vector<size_t>& pred_remain_queries) const {
    return pred_reach_probs;
  }
};
// 反事实后续节点的到达概率设置为1
struct CounterfactualSuccessors {
  std::pair<std::vector<double>, std::vector<size_t>> SuccessorReachProbs(
      const std::vector<double>& pred_reach_probs, const double* strat, size_t num_actions,
      size_t action, const std::vector<size_t>& remaining_queries) const {
    return {{1.0}, {}};
  }
};
// CFR_In: informed CF deviation
struct ImmediateInternalSequencePredecessors : public NoExternal,
                                               public AllInternal,
                                               public CounterfactualSuccessors {};

// CFR: blind CF deviation (original CFR):
// 所有外部变换，没有内部变换，后继为恒等变换
struct ImmediateExternalSequencePredecessors : public AllExternal,
                                               public NoInternal,
                                               public CounterfactualSuccessors {};

// CFR_Ex+In: 这是一种变体,目的是测试Ex+In的效果
struct ImmediateExInSequencePredecessors : public AllExternal,
                                           public AllInternal,
                                           public CounterfactualSuccessors {};

//  恒等转换后续节点的到达概率设置为 父节点到达概率*当前动作概率，note:
//  只有一个到达概率
struct IdentitySuccessors {
  std::pair<std::vector<double>, std::vector<size_t>> SuccessorReachProbs(
      const std::vector<double>& pred_reach_probs, const double* strat, size_t num_actions,
      size_t action, const std::vector<size_t>& remaining_queries) const {
    return {{pred_reach_probs[0] * strat[action]}, {}};
  }
};

// informed action deviation
// 每个信息集的memory state数量为1，并且设定为己方使用当前策略的到达概率
struct InformedActionSequencePredecessors : public NoExternal,
                                            public AllInternal,
                                            public IdentitySuccessors {};
// blind action deviation
struct BlindActionSequencePredecessors : public AllExternal,
                                         public NoInternal,
                                         public IdentitySuccessors {};

// 任意前驱节点可执行外部转换，再连续执行外部变换。当前节点的state
// memory数量=前驱节点数量
struct BlindPartialSequenceSuccessors {
  std::pair<std::vector<double>, std::vector<size_t>> SuccessorReachProbs(
      const std::vector<double>& pred_reach_probs, const double* strat, size_t num_actions,
      size_t action, const std::vector<size_t>& remaining_queries) const {
    std::vector<double> succ_reach_probs = pred_reach_probs;  // External
    succ_reach_probs.push_back(pred_reach_probs[pred_reach_probs.size() - 1] *
                               strat[action]);  // Identity
    return {succ_reach_probs, {}};
  }
};
// BPS: blind partial sequence deviation, single-target deviations generated
// form the set of blind causal deviations
struct BlindPartialSequencePredecessors : public AllExternal,
                                          public NoInternal,
                                          public BlindPartialSequenceSuccessors {};

// CFPS: counterfactual partial sequence deviation
// 通过correlation到达目标信息集，并连续执行外部变换，在结尾执行内部变换后再re-correlation,
// 当前节点的state memory数量=前驱节点数量
struct CounterfactualPartialSequencePredecessors : public NoExternal,
                                                   public AllInternal,
                                                   public BlindPartialSequenceSuccessors {};

// CFPS_Ex+In， 用来测试Ex+In的效果
struct CounterfactualPartialSequenceExInPredecessors : public AllExternal,
                                                       public AllInternal,
                                                       public BlindPartialSequenceSuccessors {};

// Causal: 计算后继节点的state memory概率，排序（外部、内部、等价）
// 当前节点变换类型包括了等价，内部，外部，在执行非等价变换后，所有后续节点只剩下外部变换
// 还需通过后面的ExternalPredecessorReachProbs/InternalPredecessorReachProbs来过滤掉不需要的memory
// state
struct CausalSuccessors {
  std::pair<std::vector<double>, std::vector<size_t>> SuccessorReachProbs(
      const std::vector<double>& pred_reach_probs, const double* strat, size_t num_actions,
      size_t action, const std::vector<size_t>& remaining_queries) const {
    std::vector<double> succ_reach_probs = pred_reach_probs;  // External
    succ_reach_probs.reserve(pred_reach_probs.size() + num_actions);
    const double identity_seq =
        pred_reach_probs[pred_reach_probs.size() - 1];  // 进行Identity的概率放到了最后位置
    // Internal，
    // 当前节点执行了内部变换，非目标动作的概率都要保留，其前置节点仍然都是Identity
    for (size_t a = 0; a < num_actions; ++a) {
      if (a != action) {
        succ_reach_probs.push_back(identity_seq * strat[a]);
      }
    }
    succ_reach_probs.push_back(identity_seq *
                               strat[action]);  // Identity：将执行当前动作的概率放在最后位置
    return {succ_reach_probs, {}};
  }
};
// CSPS: Causal Partial Sequence Predecessors
// 外部转换会记录所有前驱节点的非直达推荐的memory state概率，共有(n-1)*d种
// 内部转换会记录直达推荐的memory state概率,共有1种
struct CausalPartialSequencePredecessors : public CausalSuccessors {
  // 当节点执行外部变换，该节点可通过内部或外部变换到达，复制父节点的memory
  // state除了最后一个元素（等价变换）外的所有元素
  std::vector<double> ExternalPredecessorReachProbs(
      const std::vector<double>& pred_reach_probs,
      const std::vector<size_t>& pred_remain_queries) const {
    std::vector<double> p(pred_reach_probs.size() - 1);
    std::copy(pred_reach_probs.begin(), pred_reach_probs.end() - 1,
              p.begin());  // 复制除了最后一个元素外的所有元素
    return p;
  }
  // 当节点执行内部变换，仅通过等价变换到达，复制最后一个元素
  std::vector<double> InternalPredecessorReachProbs(
      const std::vector<double>& pred_reach_probs,
      const std::vector<size_t>& pred_remain_queries) const {
    return {pred_reach_probs[pred_reach_probs.size() - 1]};
  }
};
// TICS: twice-informed causal sequence deviation
// TICS:
// 通过等价转换到达目标信息集，并连续执行反事实外部变换强制到达结束节点，在结尾执行内部变换
// 因此，内部变换的state memory数量包含了前述的所有情况，通过没有外部变换
struct TwiceInformedPartialSequencePredecessors : public NoExternal,
                                                  public AllInternal,
                                                  public CausalSuccessors {};
// TICS-Ex+In 仅供测试
struct TwiceInformedPartialSequenceExInPredecessors : public AllExternal,
                                                      public AllInternal,
                                                      public CausalSuccessors {};
// Behavioral deviation, the most complex
struct BehavioralPredecessors : public NoExternal, public AllInternal {
  std::pair<std::vector<double>, std::vector<size_t>> SuccessorReachProbs(
      const std::vector<double>& pred_reach_probs, const double* strat, size_t num_actions,
      size_t action, const std::vector<size_t>& remaining_queries) const {
    std::vector<double> succ_reach_probs;
    succ_reach_probs.reserve(pred_reach_probs.size() * num_actions);  // 乘
    for (auto p : pred_reach_probs) {
      for (size_t a = 0; a < num_actions; ++a) {
        succ_reach_probs.push_back(p * strat[a]);  // 到达概率乘以推荐动作的概率
      }
    }
    return {succ_reach_probs, {}};
  }
};

// default external
// k_queries: 有请求次数k_queries的budget 可请求
// sequential: 是否序列：请求需要是连续的
// causal: 是否因果：请求的后续节点只能通过外部变换到达
// 采用覆盖式的设计
// external
// internal
// 不限制最大利用请求次数
struct UncompressedSCDPredecessors : public AllExternal {
  // 初始化参数的构造函数
  explicit UncompressedSCDPredecessors(size_t scd_budget, bool sequential, bool causal)
      : sequential_(sequential), causal_(causal), scd_budget_(scd_budget) {}

  // 可请求,返回具有请求次数的前驱节点的到达概率，当前节点的state
  // memory数量=前驱节点数量
  std::vector<double> InternalPredecessorReachProbs(
      const std::vector<double>& pred_reach_probs,
      const std::vector<size_t>& pred_remain_queries) const {
    std::vector<double> pred_reach_probs_with_query;
    assert(pred_reach_probs.size() == pred_remain_queries.size());
    for (size_t i = 0; i < pred_reach_probs.size(); ++i) {
      if (pred_remain_queries[i] > 0) {  // 如果剩余查询次数大于0，保留该前驱节点的到达概率
        pred_reach_probs_with_query.push_back(pred_reach_probs[i]);
      }
    }
    return pred_reach_probs_with_query;
  }

  // 计算后继节点的到达概率与剩余查询次数
  // 计算后继节点的到达概率与次数
  std::pair<std::vector<double>, std::vector<size_t>> SuccessorReachProbs(
      const std::vector<double>& pred_reach_probs, const double* strat, size_t num_actions,
      size_t action, const std::vector<size_t>& pred_remain_queries) const {
    assert(pred_reach_probs.size() == pred_remain_queries.size());
    // 不请求, 等价选取不请求选项
    std::vector<double> succ_reach_probs = pred_reach_probs;        // External
    std::vector<size_t> succ_remain_queries = pred_remain_queries;  // 请求次数不变
    // k-query: 只考虑次数
    if (!sequential_ && !causal_) {
      // 请求，对请求>0的state memory，乘以推荐动作的概率，并且查询次数减1；
      for (size_t i = 0; i < pred_reach_probs.size(); ++i) {
        if (pred_remain_queries[i] > 0) {
          // 请求，并依次获得相应的推荐
          for (size_t a = 0; a < num_actions; ++a) {
            succ_reach_probs.push_back(pred_reach_probs[i] *
                                       strat[a]);  // 到达概率乘以推荐动作的概率
            succ_remain_queries.push_back(pred_remain_queries[i] - 1);  // 查询次数减1
          }
        }
      }
      return {succ_reach_probs, succ_remain_queries};
    }

    // sequential： 查询需要连续
    if (sequential_ && !causal_) {
      // 非请求到达, 对于非连续请求情况的剩余请求次数清0
      // 判断标准为请求次数是否等于初始查询次数
      for (size_t i = 0; i < pred_remain_queries.size(); ++i) {
        // 如果前驱节点的剩余查询次数小于初始查询次数，说明已经查询过了
        if (pred_remain_queries[i] < scd_budget_) {
          succ_remain_queries[i] = 0;
        }
      }
      // 请求到达，对请求次数>0的g，请求并获得相应的推荐
      for (size_t i = 0; i < pred_remain_queries.size(); ++i) {
        if (pred_remain_queries[i] > 0) {
          for (size_t a = 0; a < num_actions; ++a) {
            succ_reach_probs.push_back(pred_reach_probs[i] *
                                       strat[a]);  // 到达概率乘以非推荐动作的概率
            succ_remain_queries.push_back(pred_remain_queries[i] - 1);  // 非推荐动作查询次数-1
          }
        }
      }
      return {succ_reach_probs, succ_remain_queries};
    }
    // causal
    if (!sequential_ && causal_) {
      // 请求到达，对偏离推荐的请求次数清0
      for (size_t i = 0; i < pred_remain_queries.size(); ++i) {
        if (pred_remain_queries[i] > 0) {
          // 请求但偏离推荐到达
          for (size_t a = 0; a < num_actions; ++a) {
            if (a != action) {
              succ_reach_probs.push_back(pred_reach_probs[i] *
                                         strat[a]);  // 到达概率乘以非推荐动作的概率
              succ_remain_queries.push_back(0);      // 非推荐动作查询次数直接清0
            }
          }
          // 请求并遵循推荐到达
          succ_reach_probs.push_back(pred_reach_probs[i] *
                                     strat[action]);                  // 到达概率乘以推荐动作的概率
          succ_remain_queries.push_back(pred_remain_queries[i] - 1);  // 查询次数减1
        }
      }
      return {succ_reach_probs, succ_remain_queries};
    }

    // sequential + causal
    if (sequential_ && causal_) {
      // 非请求到达，对于已请求过的g剩余请求次数清0
      // 判断标准：请求次数小于预算
      for (size_t i = 0; i < pred_remain_queries.size(); ++i) {
        if (pred_remain_queries[i] <
            scd_budget_) {  // 如果前驱节点的剩余查询次数小于初始查询次数，说明已经查询过了
          succ_remain_queries[i] = 0;
        }
      }
      // 请求到达，满足sequential和causal的要求
      for (size_t i = 0; i < pred_remain_queries.size(); ++i) {
        if (pred_remain_queries[i] > 0) {
          // 请求但偏离推荐到达，请求次数清0
          for (size_t a = 0; a < num_actions; ++a) {
            if (a != action) {
              succ_reach_probs.push_back(pred_reach_probs[i] *
                                         strat[a]);  // 到达概率乘以非推荐动作的概率
              succ_remain_queries.push_back(0);      // 非推荐动作查询次数直接清0
            }
          }
          // 遵循推荐到达，查询后次数减1
          succ_reach_probs.push_back(pred_reach_probs[i] *
                                     strat[action]);                  // 到达概率乘以推荐动作的概率
          succ_remain_queries.push_back(pred_remain_queries[i] - 1);  // 查询次数减1
        }
      }
      return {succ_reach_probs, succ_remain_queries};
    }

    return {pred_reach_probs, pred_remain_queries};
  }

  const size_t scd_budget_;
  const bool sequential_;
  const bool causal_;
};

// 最大效率利用请求次数。
// 压缩 SCD 结构下，由于后续节点采用了recorrelation，并不会考虑未来的请求，
// 因此，未提高算法的计算效率，当前节点有多少次请求就用多少次。
struct SCDPredecessors : UncompressedSCDPredecessors {
  // 初始化参数的构造函数
  explicit SCDPredecessors(size_t scd_budget, bool sequential, bool causal)
      : UncompressedSCDPredecessors(scd_budget, sequential, causal) {}

  // blind: 输出可请求次数为0的前驱节点概率
  std::vector<double> ExternalPredecessorReachProbs(
      const std::vector<double>& pred_reach_probs,
      const std::vector<size_t>& pred_remain_queries) const {
    std::vector<double> pred_reach_probs_with_max_query;
    assert(pred_reach_probs.size() == pred_remain_queries.size());
    // 检查是否最大利用了请求次数，如果有前驱节点的剩余请求次数等于初始查询次数，说明没有使用过请求，可以使用该前驱节点的到达概率
    for (size_t i = 0; i < pred_reach_probs.size(); ++i) {
      if (pred_remain_queries[i] ==
          0) {  // 如果前驱节点的剩余请求次数等于初始查询次数，说明没有使用过请求，可以使用该前驱节点的到达概率
        pred_reach_probs_with_max_query.push_back(pred_reach_probs[i]);
      }
    }
    return pred_reach_probs_with_max_query;
  }

  // 可请求,返回具有请求次数的前驱节点的到达概率，当前节点的state
  // memory数量=前驱节点数量
  std::vector<double> InternalPredecessorReachProbs(
      const std::vector<double>& pred_reach_probs,
      const std::vector<size_t>& pred_remain_queries) const {
    std::vector<double> pred_reach_probs_with_max_query;
    assert(pred_reach_probs.size() == pred_remain_queries.size());
    // 分别处理不同的 SCD 结构
    //  SCD or SCD_SEQ: 只输出最小剩余请求次数的前驱节点的到达概率
    if (!causal_) {
      // Exhausted memories cannot query here. Select the smallest remaining
      // positive budget that is actually present (it need not be 1).
      size_t min_remain_query = 0;
      for (size_t remaining : pred_remain_queries) {
        if (remaining > 0 && (min_remain_query == 0 || remaining < min_remain_query)) {
          min_remain_query = remaining;
        }
      }
      for (size_t i = 0; i < pred_reach_probs.size(); ++i) {
        if (min_remain_query > 0 && pred_remain_queries[i] ==
            min_remain_query) {  // 仅保留还能请求的最小正数剩余次数
          pred_reach_probs_with_max_query.push_back(pred_reach_probs[i]);
        }
      }
      return pred_reach_probs_with_max_query;
    }

    // SCD_CS or SCD_SEQ+CS:
    // 只输出剩余请求次数非0的前驱节点的到达概率，这里不限制前驱节点必须请求必遵循的限制，
    // 如果只取最小值，则要求了前驱节点都最大使用了请求次数，并遵循推荐到达。
    // todo: 但是也可以测试只取最小值的情况，看看是否能提高效率
    if (causal_) {
      for (size_t i = 0; i < pred_reach_probs.size(); ++i) {
        if (pred_remain_queries[i] > 0) {  // 如果剩余查询次数大于0，保留该前驱节点的到达概率
          pred_reach_probs_with_max_query.push_back(pred_reach_probs[i]);
        }
      }
      return pred_reach_probs_with_max_query;
    }
    return {};
  }
};

// 抽象类 继承至策略policy, 提供创建learner, 更新策略的接口Update
class CfValueTreeLearner;
using CfValueTreeLearnerPtr = std::unique_ptr<CfValueTreeLearner>;
class CfValueTreeLearner : public virtual Policy {
 public:
  template <class Subclass, class... Args>
  static std::vector<CfValueTreeLearnerPtr>
  NewList(  // 创建一个包含学习者的向量，包含每个玩家 CfValueTreeLearnerPtr
      size_t num_players, Args&&... args) {
    std::vector<CfValueTreeLearnerPtr> learners;
    learners.reserve(num_players);  // 预留空间
    for (size_t i = 0; i < num_players; ++i) {
      learners.emplace_back(new Subclass(std::forward<Args>(args)...));  // 完美转发构造参数
    }  // args... 是Subclass的构造函数参数， new class()后返回该类实例的指针
       // args: DeviationSequencePredecessors(), RmUpdate, RmLink
    return learners;
  }

 public:
  virtual ~CfValueTreeLearner() = default;
  // 更新策略函数接口
  virtual void Update(const std::vector<std::string>& initial_info_states,
                      const InfoStateUvm<CfValueTreeNode>& cf_value_tree, double epsilon = 0.0) = 0;
  // 克隆函数接口，返回一个指向自身类的智能指针
  virtual CfValueTreeLearnerPtr Clone() const = 0;
  // initial_info_states 初始节点的兄弟节点iss
  // virtual std::tuple<double, double, double, double> LearnerReachedPhiRegret(
  //     const std::vector<std::string>& initial_info_states,
  //     const InfoStateUvm<CfValueTreeNode>& cf_value_tree) const {
  //   return {0.0, 0.0, 0.0, 0.0};
  // }
  virtual std::tuple<double, double, double, double> PhiRegrets() const { return {0.0, 0.0, 0.0, 0.0}; }
  virtual StageRegretMetrics StageMetrics(double w_numerical_eps) const { return {}; }
  virtual void ResetStageEvaluation() {}
  virtual void ResetSolverRegrets() {}
  virtual void SetCfrPlus(bool enabled) {}
  virtual double SolverRegretL1() const { return 0.0; }
  virtual double StageOriginalRegretL1() const { return 0.0; }
  virtual double StageOriginalLocalRegret() const {
    return std::numeric_limits<double>::quiet_NaN();
  }
  virtual double StageExposureSum() const { return 0.0; }
};

// TabularResponder是一个策略类，继承自Policy接口
// 它使用一个InfoStateUvm<T>来存储信息状态和对应的决策信息
template <class T>  // 存储决策信息类型 ImmediateDecisionInfo
class TabularResponder : public virtual Policy {
 public:
  TabularResponder() : local_info_map_() {}
  virtual ~TabularResponder() = default;

  // 实现Policy接口的Response函数
  // 根据当前状态的合法动作，返回一个动作概率分布
  std::vector<double> Response(const open_spiel::State& state) const override final {
    const std::string info_state = state.InformationStateString();  // 获取信息状态字符串 key 值
    if (local_info_map_.contains(info_state)) {
      return local_info_map_.at(info_state).Response();
    } else {
      const int n = state.LegalActions().size();
      return std::vector<double>(n, 1.0 / n);
    }
  }

  // 获取平均策略 policy重载函数
  std::vector<double> AverageResponse(const open_spiel::State& state) const {
    const std::string info_state = state.InformationStateString();  // 获取信息状态字符串 key 值
    if (local_info_map_.contains(info_state)) {
      return local_info_map_.at(info_state).AveragePolicy();
    } else {
      const int n = state.LegalActions().size();
      return std::vector<double>(n, 1.0 / n);
    }
  }

 protected:
  InfoStateUvm<T> local_info_map_;  // hash_map table结构, key: iss，value:决策信息T
                                    // 指定ImmediateDecisionInfo
};

// 基于cfr的行为偏差学习器，定义了策略更新方法Update
template <class DeviationSequencePredecessors>
// Concept DeviationSequencePredecessors requires
// virtual std::vector<double> ExternalPredecessorReachProbs(
//     const std::vector<double>& pred_reach_probs) const;
// virtual std::vector<double> InternalPredecessorReachProbs(
//     const std::vector<double>& pred_reach_probs) const;
// virtual std::vector<double> SuccessorReachProbs(
//     const std::vector<double>& pred_reach_probs, const double* strat,
//     size_t num_actions, size_t action) const;
class BehavioralDeviationTabularCfvLearner
    : public CfValueTreeLearner,                             // 抽象类，创建learner列表,
                                                             // 更新策略的接口Update
      public TabularResponder<_tl::ImmediateDecisionInfo> {  // 定义Response函数，存储了iss:
                                                             // decision_info
                                                             // 信息状态和对应的决策信息
 public:
  template <class... Args>  // 变参模板参数，可能是0到任意个，不需要明确指定，编译会自动推导
  // 静态成员函数: 可以在没有实例化类的情况下直接通过类名调用的
  // 构造了一个自身类BehavioralDeviationTabularCfvLearner<指定模板类型>指针的列表
  // 默认行为：传入 update_target, f
  static std::vector<CfValueTreeLearnerPtr> NewList(size_t num_players,
                                                    _tl::RegretTransformation update_target,
                                                    _tl::LinkFn f) {
    return CfValueTreeLearner::NewList<BehavioralDeviationTabularCfvLearner>(
        num_players, DeviationSequencePredecessors(), update_target, f);
  }

  // 重载：为像 SCDPredecessors 这样的类型，允许把 k/seq/caus
  // 传给前驱类型的构造函数
  static std::vector<CfValueTreeLearnerPtr> NewList(size_t num_players,
                                                    _tl::RegretTransformation update_target,
                                                    _tl::LinkFn f, size_t scd_budget,
                                                    bool sequential = false, bool causal = false) {
    return CfValueTreeLearner::NewList<BehavioralDeviationTabularCfvLearner>(
        num_players, DeviationSequencePredecessors(scd_budget, sequential, causal), update_target,
        f, scd_budget, sequential, causal);
  }

 private:
  // 栈状态 结构体，存储iss和对应的reach_prob_lists
  struct StackState {
    const std::string* info_state_string_;         // 信息状态字符串指针
    const _tl::ReachProbLists* reach_prob_lists_;  // phi的权重：未偏离/遵循推荐时的前驱概率列表
    // const double own_reach_prob; //
    // 当前节点的己方到达概率，用于计算序列策略概率

    void Reset(const std::string* info_state_string, const _tl::ReachProbLists* reach_prob_lists) {
      info_state_string_ = info_state_string;
      reach_prob_lists_ = reach_prob_lists;
      // own_reach_prob = own_reach_prob;
    }
  };

 public:
  // 构造函数
  BehavioralDeviationTabularCfvLearner(DeviationSequencePredecessors dev_seq_predecessors,
                                       const _tl::RegretTransformation& update_target,
                                       const _tl::LinkFn& f, size_t scd_budget = 0,
                                       bool sequential = false, bool causal = false)
      : TabularResponder(),  // 创建一个TabularResponder实例，空的local_info_map_
        update_target_(update_target),
        f_(f),
        dev_seq_predecessors_(std::move(dev_seq_predecessors)),
        scd_budget_(scd_budget),
        sequential_(sequential),
        causal_(causal) {}
  virtual ~BehavioralDeviationTabularCfvLearner() = default;

  // 更新策略函数，接收初始节点{iss}和cf_value_tree,更新的顺序是自顶向下-深度搜索
  void Update(const std::vector<std::string>&
                  initial_info_states,  // initial_info_states 初始节点的兄弟节点iss
              const InfoStateUvm<CfValueTreeNode>& cf_value_tree,
              double epsilon = 0.0) override final {
    std::vector<StackState> state_stack(
        cf_value_tree.size());  // 栈，存储当前iss和对应的reach_prob_lists的指针，全部初始化为空指针
    const auto initial_reach_prob_lists = new _tl::ReachProbLists{
        {1.0}, {1.0}, {scd_budget_}};  // 初始化前驱概率列表，常量，prev_和next_都为{1.0}
    int stack_idx = 0;
    // 将所有初始 initial_info_states
    // 入栈，初始化栈所有节点为iss和reach_prob_lists，其到达概率为1
    for (; stack_idx < initial_info_states.size(); ++stack_idx) {
      state_stack[stack_idx].Reset(&(initial_info_states[stack_idx]), initial_reach_prob_lists);
    }
    --stack_idx;  // 指向栈顶的最后一个根节点（之前多+1）
    std::vector<_tl::ReachProbLists*> all_reach_prob_lists = {
        initial_reach_prob_lists};  // 初始化到达概率列表的向量，用于后面释放指针
    all_reach_prob_lists.reserve(cf_value_tree.size());  // 并预留空间，避免频繁扩容
    // 开始策略更新
    while (stack_idx >= 0) {
      const auto [info_state_string_ptr, pred_reach_probs] =
          state_stack[stack_idx];  // 获取栈顶的iss和ReachProbLists,
                                   // 己方到达概率： 用更新前的还是更新后的？
      --stack_idx;                 // 出栈

      const auto& [cf_values, child_keys, iwrp] =
          cf_value_tree.at(*info_state_string_ptr);  // 获取当前iss对应的cf_values和child_keys,
                                                     // iwrp
      const size_t num_actions = cf_values.Size();   // 当前iss的动作数量

      // 只有一个动作，省略后面的策略更新步骤，直接更新子节点的reach_prob_lists，并将stack_idx指向子节点
      if (num_actions < 2) {
        for (size_t child_idx = 0; child_idx < child_keys[0].size(); ++child_idx) {
          ++stack_idx;  // 子兄弟节点依次入栈
          state_stack[stack_idx].Reset(
              &(child_keys[0][child_idx]),  // 填入iss和当前节点的reach_prob_lists
              pred_reach_probs);
        }
        continue;
      }
      // 获取外部转换的到达概率 w^t,w^{t+1}
      const _tl::ReachProbLists ex_reach_probs = {
          dev_seq_predecessors_.ExternalPredecessorReachProbs(pred_reach_probs->prev_,
                                                              pred_reach_probs->remaining_queries_),
          dev_seq_predecessors_.ExternalPredecessorReachProbs(
              pred_reach_probs->next_, pred_reach_probs->remaining_queries_)};
      // 获取内部转换的到达概率 w^t,w^{t+1}
      // 对于query类型，内部转换需要请求，要求可请求次数>0
      const _tl::ReachProbLists in_reach_probs = {
          dev_seq_predecessors_.InternalPredecessorReachProbs(pred_reach_probs->prev_,
                                                              pred_reach_probs->remaining_queries_),
          dev_seq_predecessors_.InternalPredecessorReachProbs(
              pred_reach_probs->next_, pred_reach_probs->remaining_queries_)};

      // debug：打印外部转换和内部转换的到达概率
      if (false && t < 10 && infoset < 10) {
        ++infoset;
        std::cout << infoset << "| ex_reach_probs prev: ";
        for (const auto& p : ex_reach_probs.prev_) {
          std::cout << p << " ";
        }
        std::cout << "ex_reach_probs next: ";
        for (const auto& p : ex_reach_probs.next_) {
          std::cout << p << " ";
        }
        std::cout << "in_reach_probs prev: ";
        for (const auto& p : in_reach_probs.prev_) {
          std::cout << p << " ";
        }
        std::cout << "in_reach_probs next: ";
        for (const auto& p : in_reach_probs.next_) {
          std::cout << p << " ";
        }
        std::cout << std::endl;
      }
      // auto=ImmediateDecisionInfo
      // 获取或创建当前iss对应的决策信息，包含了当前iss的策略、遗憾值等信息
      // 该类继承了TabularResponder<_tl::ImmediateDecisionInfo>
      // 用来更新策略，对应玩家创造决策节点
      auto& local_decision_info =
          GetOrCreateFromArgs(local_info_map_, *info_state_string_ptr, num_actions,
                              ex_reach_probs.Size(), in_reach_probs.Size());
      local_decision_info.SetCfrPlus(is_cfr_plus_);

      double prev_policy[num_actions];            // 存储当前策略
      local_decision_info.Response(prev_policy);  // 获取当前策略,并存储在prev_policy中
      // 更新iss的策略：计算phi的内外部遗憾值，并更新策略
      local_decision_info.Update(cf_values, update_target_, f_, ex_reach_probs, in_reach_probs,
                                 epsilon, iwrp);

      double next_policy[num_actions];            // 存储更新后的策略
      local_decision_info.Response(next_policy);  // 获取更新后的策略,并存储在next_policy中

      // 更新子节点的reach_prob_lists
      for (size_t a = 0; a < num_actions; ++a) {
        if (child_keys[a].size() < 1) {
          continue;
        }
        // 计算每个动作a的后继己方节点的到达概率列表 w^t,w^{t+1}
        // 区分查询类型和非查询类型，前者需要传入剩余查询次数，后者不需要
        // 查询的SuccessorReachProbs会返回每个前驱节点对应的所有后继节点的到达概率和剩余查询次数，而非查询类型的SuccessorReachProbs只返回到达概率
        // _tl::ReachProbLists* successor_reach_prob_lists; //
        // 定义指针，指向为每个子节点创建的reach_prob_lists
        const auto [succ_reach_probs, succ_remain_queries] =
            dev_seq_predecessors_.SuccessorReachProbs(pred_reach_probs->prev_, prev_policy,
                                                      num_actions, a,
                                                      pred_reach_probs->remaining_queries_);
        const auto [succ_reach_probs_next, succ_remain_queries_next] =
            dev_seq_predecessors_.SuccessorReachProbs(pred_reach_probs->next_, next_policy,
                                                      num_actions, a,
                                                      pred_reach_probs->remaining_queries_);
        // 为每个子节点创建新的reach_prob_lists,
        // 包含了到达概率和剩余查询次数,查询次数不依赖策略
        const auto successor_reach_prob_lists =
            new _tl::ReachProbLists{succ_reach_probs, succ_reach_probs_next, succ_remain_queries};

        // 将所有子节点入栈，更新栈顶为子节点iss和对应的reach_prob_lists，实现从根节点到叶节点的遍历更新策略
        for (size_t child_idx = 0; child_idx < child_keys[a].size(); ++child_idx) {
          ++stack_idx;
          state_stack[stack_idx].Reset(&(child_keys[a][child_idx]), successor_reach_prob_lists);
        }
        all_reach_prob_lists.push_back(successor_reach_prob_lists);
      }
    }
    // 一轮迭代后，清理所有动态分配的reach_prob_lists内存
    if (debug_print_) {
      // debug，打印输出all_reach_prob_lists的大小和每个ptr指向的reach_prob_lists的内容
      for (size_t i = 0; i < all_reach_prob_lists.size(); ++i) {
        // std::cout << "all_reach_prob_lists size: " <<
        // all_reach_prob_lists.size() << std::endl;
        auto ptr = all_reach_prob_lists[i];
        std::cout << "reach_prob_lists :" << i << " content: " << std::endl;
        std::cout << "prev: ";
        for (const auto& p : ptr->prev_) {
          std::cout << p << " ";
        }
        std::cout << "next: ";
        for (const auto& p : ptr->next_) {
          std::cout << p << " ";
        }
        std::cout << "remaining_queries: ";
        for (const auto& q : ptr->remaining_queries_) {
          std::cout << q << " ";
        }
        std::cout << std::endl;
      }
      // debug_print_ = false;
    }
    for (auto ptr : all_reach_prob_lists) {
      delete ptr;
    }
    ++t;
    infoset = 0;
  }

  // 克隆当前学习者，返回一个新的自身实例的指针
  CfValueTreeLearnerPtr Clone() const override final {
    return CfValueTreeLearnerPtr(new BehavioralDeviationTabularCfvLearner(*this));  // *this
  }
  // 获取总计和max phi遗憾值
  // std::tuple<double, double, double, double> LearnerReachedPhiRegret(
  //     const std::vector<std::string>&
  //         initial_info_states,  // initial_info_states 初始节点的兄弟节点iss
  //     const InfoStateUvm<CfValueTreeNode>& cf_value_tree) const override final {
  //   // 每轮初始化最大phi-regret
  //   double global_max_full_phi_regret_fus = 0.0;
  //   double global_max_full_phi_regret = 0.0;
  //   double global_max_phi_regret = 0.0;
  //   double global_sum_phi_regret = 0.0;
  //   std::vector<StackState> state_stack(
  //       cf_value_tree.size());  // 栈，存储当前iss和对应的reach_prob_lists的指针，全部初始化为空指针
  //   const auto initial_reach_prob_lists =
  //       new _tl::ReachProbLists{{1.0}, {1.0}};  // 初始化前驱概率列表，常量，prev_和next_都为{1.0}
  //   int stack_idx = 0;
  //   // 将所有初始 initial_info_states
  //   // 入栈，初始化栈所有节点为iss和reach_prob_lists，其到达概率为1
  //   for (; stack_idx < initial_info_states.size(); ++stack_idx) {
  //     state_stack[stack_idx].Reset(&(initial_info_states[stack_idx]), initial_reach_prob_lists);
  //   }
  //   --stack_idx;  // 指向栈顶的最后一个根节点（之前多+1）
  //   std::vector<_tl::ReachProbLists*> all_reach_prob_lists = {
  //       initial_reach_prob_lists};  // 初始化到达概率列表的向量，用于后面释放指针
  //   all_reach_prob_lists.reserve(cf_value_tree.size());  // 并预留空间，避免频繁扩容
  //   // 开始策略更新
  //   while (stack_idx >= 0) {
  //     const auto [info_state_string_ptr, pred_reach_probs] =
  //         state_stack[stack_idx];  // 获取栈顶的iss和ReachProbLists,
  //                                  // 己方到达概率： 用更新前的还是更新后的？
  //     --stack_idx;                 // 出栈

  //     const auto& [cf_values, child_keys, iwrp] =
  //         cf_value_tree.at(*info_state_string_ptr);  // 获取当前iss对应的cf_values和child_keys,
  //                                                    // iwrp
  //     const size_t num_actions = cf_values.Size();   // 当前iss的动作数量

  //     // 只有一个动作，省略后面的策略更新步骤，直接更新子节点的reach_prob_lists，并将stack_idx指向子节点
  //     if (num_actions < 2) {
  //       for (size_t child_idx = 0; child_idx < child_keys[0].size(); ++child_idx) {
  //         ++stack_idx;  // 子兄弟节点依次入栈
  //         state_stack[stack_idx].Reset(
  //             &(child_keys[0][child_idx]),  // 填入iss和当前节点的reach_prob_lists
  //             pred_reach_probs);
  //       }
  //       continue;
  //     }
  //     // 获取外部转换的到达概率 w^t,w^{t+1}
  //     const _tl::ReachProbLists ex_reach_probs = {
  //         dev_seq_predecessors_.ExternalPredecessorReachProbs(pred_reach_probs->prev_,
  //                                                             pred_reach_probs->remaining_queries_),
  //         dev_seq_predecessors_.ExternalPredecessorReachProbs(
  //             pred_reach_probs->next_, pred_reach_probs->remaining_queries_)};
  //     // 获取内部转换的到达概率 w^t,w^{t+1}
  //     const _tl::ReachProbLists in_reach_probs = {
  //         dev_seq_predecessors_.InternalPredecessorReachProbs(pred_reach_probs->prev_,
  //                                                             pred_reach_probs->remaining_queries_),
  //         dev_seq_predecessors_.InternalPredecessorReachProbs(
  //             pred_reach_probs->next_, pred_reach_probs->remaining_queries_)};
  //     // auto=ImmediateDecisionInfo
  //     // 获取或创建当前iss对应的决策信息，包含了当前iss的策略、遗憾值等信息
  //     // 该类继承了TabularResponder<_tl::ImmediateDecisionInfo>
  //     // 用来更新策略，对应玩家创造决策节点
  //     auto& local_decision_info =
  //         local_info_map_.at(*info_state_string_ptr);  // 获取当前iss对应的决策信息

  //     // 获取局部 node的 max phi-regret
  //     // const std::vector<double>& local_policy =
  //     // policy.GetPolicy(*info_state_string_ptr); // 获取当前策略,
  //     // policy是MapPolicy类 // TODO 这里需指定是MapPolicy类
  //     auto [max_full_phi_regret_fus, max_full_phi_regret, max_phi_regret, sum_phi_regret] =
  //         local_decision_info.PhiRegret_(cf_values, ex_reach_probs, in_reach_probs,
  //                                        iwrp);  // 计算当前节点的总计phi遗憾值和最大phi遗憾值 //
  //                                                // TODO debug 测试在扰动空间中的phi regret优化效果
  //     if (max_full_phi_regret_fus > global_max_full_phi_regret_fus) {
  //       global_max_full_phi_regret_fus = max_full_phi_regret_fus;
  //     }
  //     if (max_full_phi_regret > global_max_full_phi_regret) {
  //       global_max_full_phi_regret = max_full_phi_regret;
  //     }
  //     // 更新全局 max_reached_phi_regret
  //     if (max_phi_regret > global_max_phi_regret) {
  //       global_max_phi_regret = max_phi_regret;
  //     }
  //     global_sum_phi_regret += sum_phi_regret;

  //     // double next_policy[num_actions]; // 存储更新后的策略
  //     // local_decision_info.Response(next_policy); //
  //     // 获取更新后的策略,并存储在next_policy中
  //     double prev_policy[num_actions];            // 存储当前策略
  //     local_decision_info.Response(prev_policy);  // 获取当前策略,并存储在prev_policy中

  //     // 更新子节点的reach_prob_lists
  //     for (size_t a = 0; a < num_actions; ++a) {
  //       if (child_keys[a].size() < 1) {
  //         continue;
  //       }
  //       // 计算每个动作a的后继己方节点的到达概率列表 w^t,w^{t+1}
  //       // _tl::ReachProbLists* successor_reach_prob_lists;
  //       // 如果采用SCDPredecessors，则使用5参数版本，在编译
  //       const auto [succ_reach_probs, succ_remain_queries] =
  //           dev_seq_predecessors_.SuccessorReachProbs(pred_reach_probs->prev_, prev_policy,
  //                                                     num_actions, a,
  //                                                     pred_reach_probs->remaining_queries_);
  //       // const auto [succ_reach_probs_next, succ_remain_queries_next] =
  //       //     dev_seq_predecessors_.SuccessorReachProbs(
  //       //       pred_reach_probs->next_, local_policy.data(), num_actions,
  //       //       a, pred_reach_probs->remaining_queries_);
  //       const auto successor_reach_prob_lists =
  //           new _tl::ReachProbLists{succ_reach_probs, succ_reach_probs, succ_remain_queries};

  //       // 将所有子节点入栈，更新栈顶为子节点iss和对应的reach_prob_lists，实现从根节点到叶节点的遍历更新策略
  //       for (size_t child_idx = 0; child_idx < child_keys[a].size(); ++child_idx) {
  //         ++stack_idx;
  //         state_stack[stack_idx].Reset(&(child_keys[a][child_idx]), successor_reach_prob_lists);
  //       }
  //       all_reach_prob_lists.push_back(successor_reach_prob_lists);  // 保存指针,用于后续释放
  //     }
  //   }
  //   // 一轮迭代后，清理所有动态分配的reach_prob_lists内存
  //   for (auto ptr : all_reach_prob_lists) {
  //     delete ptr;
  //   }
  //   return {global_max_full_phi_regret_fus, global_max_full_phi_regret, global_max_phi_regret, global_sum_phi_regret};
  // }

  // 获取所有信息集下的最大遗憾值和总计遗憾值
  std::tuple<double, double, double, double> PhiRegrets() const override final {
    double global_max_full_phi_regret_fus = 0.0;
    double global_max_full_phi_regret = 0.0;
    double global_max_phi_regret = 0.0;
    double global_sum_phi_regret = 0.0;
    for (const auto& [info_state_string, local_decision_info] : local_info_map_) {
      auto [max_full_phi_regret_fus, max_full_phi_regret, max_phi_regret, sum_phi_regret] = local_decision_info.PhiRegrets();
      if (max_full_phi_regret_fus > global_max_full_phi_regret_fus) {
        global_max_full_phi_regret_fus = max_full_phi_regret_fus;
      }
      if (max_full_phi_regret > global_max_full_phi_regret) {
        global_max_full_phi_regret = max_full_phi_regret;
      }
      if (max_phi_regret > global_max_phi_regret) {
        global_max_phi_regret = max_phi_regret;
      }
      global_sum_phi_regret += sum_phi_regret;
    }
    return {global_max_full_phi_regret_fus, global_max_full_phi_regret, global_max_phi_regret, global_sum_phi_regret};
  }

  double StageOriginalLocalRegret() const override final {
    double maximum = 0.0;
    for (const auto& entry : local_info_map_) {
      maximum = std::max(maximum, entry.second.StageOriginalLocalRegret());
    }
    return maximum;
  }

  StageRegretMetrics StageMetrics(double w_numerical_eps) const override final {
    StageRegretMetrics metrics;
    for (const auto& entry : local_info_map_) {
      auto local_metrics = entry.second.StageMetrics(w_numerical_eps);
      for (auto& exposure : local_metrics.exposure_entries) {
        exposure.info_state = entry.first;
      }
      if (local_metrics.has_bottleneck) {
        local_metrics.bottleneck_info_state = entry.first;
      }
      metrics.Merge(local_metrics);
    }
    metrics.Finalize();
    return metrics;
  }

  void ResetStageEvaluation() override final {
    for (auto& entry : local_info_map_) entry.second.ResetStageEvaluation();
  }

  void ResetSolverRegrets() override final {
    for (auto& entry : local_info_map_) entry.second.ResetSolverRegrets();
  }

  void SetCfrPlus(bool enabled) override final {
    is_cfr_plus_ = enabled;
    for (auto& entry : local_info_map_) entry.second.SetCfrPlus(enabled);
  }

  double SolverRegretL1() const override final {
    double total = 0.0;
    for (const auto& entry : local_info_map_) total += entry.second.SolverRegretL1();
    return total;
  }

  double StageOriginalRegretL1() const override final {
    double total = 0.0;
    for (const auto& entry : local_info_map_) total += entry.second.StageOriginalRegretL1();
    return total;
  }

  double StageExposureSum() const override final {
    double total = 0.0;
    for (const auto& entry : local_info_map_) total += entry.second.StageExposureSum();
    return total;
  }

 private:
  const _tl::RegretTransformation update_target_;  // 遗憾值更新函数
  const _tl::LinkFn f_;                            // 遗憾值链接函数
  const DeviationSequencePredecessors
      dev_seq_predecessors_;  // 定义转换信息集和后继节点的到达概率state
                              // memory计算方法
  // double max_reached_phi_regret_ = 0.0; // 最大phi遗憾值
  const size_t scd_budget_;  // SCD 偏差的查询预算
  const bool sequential_;      // 是否顺序更新
  const bool causal_;          // 是否因果更新
  bool is_cfr_plus_ = true;
  // debug 打印标识
  bool debug_print_ = false;
  int t = 0;        // debug，记录迭代次数
  int infoset = 0;  // debug，记录信息集数量
};

}  // namespace hr_edl

#endif  // HR_EDL_TABULAR_LEARNER_H_
