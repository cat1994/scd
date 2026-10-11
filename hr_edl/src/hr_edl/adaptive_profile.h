#ifndef HR_EDL_ADAPTIVE_PROFILE_H_
#define HR_EDL_ADAPTIVE_PROFILE_H_

#include <algorithm>
#include <tuple>

#include "hr_edl/best_response.h"
#include "hr_edl/decision_point.h"
#include "hr_edl/policy_evaluation.h"
#include "hr_edl/samplers.h"
#include "hr_edl/tabular_learner.h"
#include "hr_edl/types.h"

namespace hr_edl {
class AdaptiveProfile;
using AdaptiveProfilePtr =
    std::unique_ptr<AdaptiveProfile>;  // 智能指针，指向AdaptiveProfile的实例，包括子类

// 抽象类 定义了算法的适应性配置文件的接口
class AdaptiveProfile {
 public:
  virtual ~AdaptiveProfile() = default;
  // 返回指定玩家的策略指针
  virtual Policy* Strategy(size_t player) const = 0;
  // 返回对抗每个对手cfv的平均值
  Cfv Ev(DecisionPoint& root, MccfrSampler& sampler, const PolicyProfile& compatriots) const {
    double avg = 0.0;
    for (int player = 0; player < root.NumPlayers(); ++player) {
      const auto [v, _] =
          PolicyValue(root, player, compatriots.WithSubstitute(Strategy(player), player), sampler);
      avg += (v - avg) / (player + 1.0);  // 计算平均值
    }
    return avg;
  }
  // 更新策略并返回期望值
  virtual Cfv UpdateAndReturnEv(DecisionPoint& root, MccfrSampler& sampler,
                                const PolicyProfile& compatriots) = 0;
  // 轮流更新并返回期望值
  virtual Cfv UpdateAlternateAndReturnEv(DecisionPoint& root, MccfrSampler& sampler) = 0;

  virtual PolicyProfile Frozen() const = 0;
  MapPolicy ToMapPolicy(DecisionPoint& root) const { return MapPolicy(Frozen(), root); }

  // 遍历博弈树，使用平均策略计算 总计 / 最大 phi遗憾值
  // virtual std::tuple<double, double, double, double> ReachedPhiRegret(DecisionPoint& root,
  //                                                                     MccfrSampler& sampler) {
  //   return {0.0, 0.0, 0.0, 0.0};
  // }
  virtual std::tuple<double, double, double, double> PhiRegrets() const {
    return {0.0, 0.0, 0.0, 0.0};
  }
  virtual void ShrinkEpsilon(double new_epsilon) {}
  virtual double Epsilon() const { return 0.0; }
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

// cfr算法配置，继承至AdaptiveProfile
class CfTreeLearnerProfile : public AdaptiveProfile {
 public:
  // learners_: 使用智能指针创建一个包含学习者的向量，包含每个玩家 CfValueTreeLearnerPtr
  // evaluators_：每个玩家对应一个PolicyCfValueTreeEvaluator，用于计算该玩家的cfv树,
  // 给定根节点、玩家策略和采样器
  CfTreeLearnerProfile(std::vector<CfValueTreeLearnerPtr>&& learners)
      : learners_(std::move(learners)), evaluators_(NewEvaluators()) {}

  Policy* Strategy(size_t player) const override final { return learners_[player].get(); }
  // 更新策略并返回期望值
  // compatriots: 玩家的策略组，替换player位置的策略为当前玩家的策略
  Cfv UpdateAndReturnEv(DecisionPoint& root, MccfrSampler& sampler,
                        const PolicyProfile& compatriots) override final {
    double avg = 0.0;
    // 对于当前玩家处于不同顺位，计算其cfv树,用于后续更新策略
    for (int player = 0; player < root.NumPlayers(); ++player) {
      // 获取需要更新的初始节点siblings的iss和cfv树
      const auto [v, initial_info_states, cf_value_tree, _] =
          evaluators_[player].ComputeCfValueTreeEvaluation(
              root, compatriots.WithSubstitute(Strategy(player), player), sampler);
      // 使用初始节点的iss和cfv树更新该玩家的策略
      learners_[player]->Update(initial_info_states, *cf_value_tree);
      // 计算所有玩家的平均值，该式为平均值公式，等价于 avg = (avg*player + v)/(player+1)
      avg += (v - avg) / (player + 1.0);
    }
    return avg;
  }
  // 轮流更新并返回期望值
  Cfv UpdateAlternateAndReturnEv(DecisionPoint& root, MccfrSampler& sampler) override final {
    double avg = 0.0;
    for (int player = 0; player < root.NumPlayers(); ++player) {
      const auto [v, initial_info_states, cf_value_tree, _] =
          evaluators_[player].ComputeCfValueTreeEvaluation(root, PolicyRefProfile(learners_),
                                                           sampler);
      learners_[player]->Update(initial_info_states, *cf_value_tree);
      avg += (v - avg) / (player + 1.0);
    }
    return avg;
  }
  // 函数的签名返回类型，等价return PolicyProfile(Clone(learners_))，隐式调用其构造函数
  PolicyProfile Frozen() const override final { return Clone(learners_); }

  // Aggregate the same cumulative phi-regret metrics used by PBCE.
  std::tuple<double, double, double, double> PhiRegrets() const override final {
    double max_full_phi_regret_fus = 0.0;
    double max_full_phi_regret = 0.0;
    double max_phi_regret = 0.0;
    double sum_phi_regret = 0.0;
    for (const auto& learner : learners_) {
      const auto [full_fus, full, conditional, sum] = learner->PhiRegrets();
      max_full_phi_regret_fus = std::max(max_full_phi_regret_fus, full_fus);
      max_full_phi_regret = std::max(max_full_phi_regret, full);
      max_phi_regret = std::max(max_phi_regret, conditional);
      sum_phi_regret += sum;
    }
    return {max_full_phi_regret_fus, max_full_phi_regret, max_phi_regret, sum_phi_regret};
  }

  double StageOriginalLocalRegret() const override final {
    double maximum = 0.0;
    for (const auto& learner : learners_) {
      const double value = learner->StageOriginalLocalRegret();
      if (!std::isfinite(value)) return value;
      maximum = std::max(maximum, value);
    }
    return maximum;
  }

  void SetCfrPlus(bool enabled) override final {
    for (auto& learner : learners_) learner->SetCfrPlus(enabled);
  }

 private:
  // 创建每个玩家对应的PolicyCfValueTreeEvaluator，将记录对应玩家的cfv树
  std::vector<PolicyCfValueTreeEvaluator> NewEvaluators() {
    std::vector<PolicyCfValueTreeEvaluator> l;
    for (size_t i = 0; i < learners_.size(); ++i) {
      l.emplace_back(i);  // = l.push_back(PolicyCfValueTreeEvaluator(i));
    }
    return l;
  }

 private:
  std::vector<CfValueTreeLearnerPtr> learners_;
  std::vector<PolicyCfValueTreeEvaluator> evaluators_;
};

// // Reward transformation cfr算法配置，继承至AdaptiveProfile
// class RTCfTreeLearnerProfile : public AdaptiveProfile {
//  public:
//   RTCfTreeLearnerProfile(std::vector<CfValueTreeLearnerPtr>&& learners, double rt_weight, size_t
//   T)
//       : learners_(std::move(learners)),
//         evaluators_(NewEvaluators()),
//         rt_policy_profile_(std::make_unique<PolicyProfile>(Frozen())),
//         iteration_(1),
//         rt_weight_(rt_weight),
//         T_(T) {}

//   Policy* Strategy(size_t player) const override final {
//     return learners_[player].get();
//   }

//   // print std::vector<double>
//   template <typename T>
//   void print_vector(const std::vector<T>& vec) const {
//     for (size_t i = 0; i < vec.size() && i < 10; ++i) {
//       std::cout << vec[i] << " ";
//     }
//     std::cout << std::endl;
//   }
//   // print template value first 10 entries of a map
//   template <typename T>
//   void print_map(const T& map) const {
//     size_t count = 0;
//     for (const auto& [key, value] : map) {
//       if (count++ < 10) {
//         std::cout << "Key: " << key << ", Value: ";
//         print_vector(value);
//       } else {
//         break;
//       }
//     }
//   }
//   Cfv UpdateAndReturnEv(DecisionPoint& root, MccfrSampler& sampler,
//                         const PolicyProfile& compatriots) override final {
//     MapPolicy policy=ToMapPolicy(root);
//     const auto& rt_value_map = ComputeRTValue(root, policy);
//     double avg = 0.0;
//     //print rt value map at second iteration
//     // if (iteration_ == 5) {
//     //   std::cout << "RT Value Map Sample (first 10 entries):" << std::endl;
//     //   print_map(rt_value_map); // debug
//     // }
//     for (int player = 0; player < root.NumPlayers(); ++player) {
//       auto [v, initial_info_states, cf_value_tree, _] =
//           evaluators_[player].ComputeCfValueTreeEvaluation(
//               root, compatriots.WithSubstitute(Strategy(player), player),
//               sampler);
//       // 计算rtcfv
//       const auto rt_cf_value_tree=ComputeRTcfValueTree(std::move(*cf_value_tree), rt_value_map);
//       learners_[player]->Update(initial_info_states, rt_cf_value_tree);
//       // 计算所有玩家的平均值，该式为平均值公式，等价于 avg = (avg*player + v)/(player+1)

//       avg += (v - avg) / (player + 1.0);
//     }
//     ++iteration_; // 迭代次数+1
//     // 每T次更新rt policy
//     if (iteration_ % T_ == 0) {
//       rt_policy_profile_ = std::make_unique<PolicyProfile>(Frozen());
//     }
//     return avg;
//   }

//   Cfv UpdateAlternateAndReturnEv(DecisionPoint& root,
//                                  MccfrSampler& sampler) override final {
//                                   MapPolicy policy=ToMapPolicy(root);
//     const auto& rt_value_map = ComputeRTValue(root, policy);
//     double avg = 0.0;

//     for (int player = 0; player < root.NumPlayers(); ++player) {
//       auto [v, initial_info_states, cf_value_tree, _] =
//           evaluators_[player].ComputeCfValueTreeEvaluation(
//               root, PolicyRefProfile(learners_), sampler);
//       // 计算rtcfv
//       const auto rt_cf_value_tree=ComputeRTcfValueTree(std::move(*cf_value_tree), rt_value_map);
//       learners_[player]->Update(initial_info_states, rt_cf_value_tree);
//       avg += (v - avg) / (player + 1.0);
//     }
//     ++iteration_;
//     // 每T次更新rt policy
//     if (iteration_ % T_ == 0) {
//       rt_policy_profile_ = std::make_unique<PolicyProfile>(Frozen());
//     }
//     return avg;
//   }

//   PolicyProfile Frozen() const override final { return Clone(learners_); }

//   //递归计算所有状态下的rt价值，返回一个InfoStateUvm<std::vector<double>>指针
//   InfoStateUvm<std::vector<double>> ComputeRTValue(DecisionPoint& root, Policy& policy) const {
//     InfoStateUvm<std::vector<double>> rt_value_map;
//     ForEachState(root, [&rt_value_map, &policy, this](const DecisionPoint& decision_point){
//       // 不执行机会节点或者终端节点
//       if (decision_point.PlayerToAct() == -1) return;
//       //get policy and rt_policy at node
//       assert(decision_point.OpenSpielStatePtr());
//       const std::vector<double> policy_state=
//       policy.Response(*decision_point.OpenSpielStatePtr()); const std::vector<double>
//       rt_policy_state= rt_policy_profile_->Response(*decision_point.OpenSpielStatePtr());
//       // compute rt_value at node
//       std::vector<double> rt_value(policy_state.size(), 0.0);
//       for (size_t aidx=0; aidx<decision_point.NumActions(); ++aidx){
//         rt_value[aidx]= rt_weight_*(rt_policy_state[aidx]- policy_state[aidx]);
//       }
//       // insert or update rt value to map
//       const std::string& iss = decision_point.InformationStateStringRef();
//       rt_value_map[iss] = std::move(rt_value);
//     }, ALL_PLAYERS);
//     return rt_value_map;
//   }

//   // 计算rtcfv树，返回一个InfoStateUvm<CfValueTreeNode>指针
//   InfoStateUvm<CfValueTreeNode> ComputeRTcfValueTree(InfoStateUvm<CfValueTreeNode> cf_value_tree,
//                                                     const InfoStateUvm<std::vector<double>>&
//                                                     rt_value_map) const {
//     auto rt_cf_value_tree = std::move(cf_value_tree); // move instead of copy
//     for (auto& [iss, node] : rt_cf_value_tree) {
//       // find rt value at iss
//       assert(rt_value_map.contains(iss));
//       const auto& rt_value = rt_value_map.at(iss);
//       assert(rt_value.size() == node.cf_values_.Size());
//       // update rt cf values at node
//       for (size_t aidx = 0; aidx < node.cf_values_.Size(); ++aidx) {
//         node.cf_values_[aidx] += rt_value[aidx];
//       }
//     }
//     return rt_cf_value_tree;
//   }
//  private:
//  // 创建每个玩家对应的PolicyCfValueTreeEvaluator，将记录对应玩家的cfv树
//   std::vector<PolicyCfValueTreeEvaluator> NewEvaluators() {
//     std::vector<PolicyCfValueTreeEvaluator> l;
//     for (size_t i = 0; i < learners_.size(); ++i) {
//       l.emplace_back(i); // = l.push_back(PolicyCfValueTreeEvaluator(i));
//     }
//     return l;
//   }
//  private:
//   std::vector<CfValueTreeLearnerPtr> learners_;
//   std::vector<PolicyCfValueTreeEvaluator> evaluators_;
//   double rt_weight_;
//   std::unique_ptr<PolicyProfile> rt_policy_profile_; // rt policy, update for each T iterations
//   size_t T_;
//   size_t iteration_;
// };

// Reward transformation cfr算法配置，继承至AdaptiveProfile
class PerturbedCfTreeLearnerProfile : public AdaptiveProfile {
 public:
  PerturbedCfTreeLearnerProfile(std::vector<CfValueTreeLearnerPtr>&& learners, double epsilon)
      : learners_(std::move(learners)),
        evaluators_(NewEvaluators()),
        // rt_policy_profile_(std::make_unique<PolicyProfile>(Frozen())),
        iteration_(1),
        epsilon_(epsilon) {}

  Policy* Strategy(size_t player) const override final { return learners_[player].get(); }

  // Policy* AverageStrategy(size_t player) const {
  //   return learners_[player].get();
  // }

  // print std::vector<double>
  template <typename T>
  void print_vector(const std::vector<T>& vec) const {
    for (size_t i = 0; i < vec.size() && i < 10; ++i) {
      std::cout << vec[i] << " ";
    }
    std::cout << std::endl;
  }
  // print template value first 10 entries of a map
  template <typename T>
  void print_map(const T& map) const {
    size_t count = 0;
    for (const auto& [key, value] : map) {
      if (count++ < 10) {
        std::cout << "Key: " << key << ", Value: ";
        print_vector(value);
      } else {
        break;
      }
    }
  }

  Cfv UpdateAndReturnEv(DecisionPoint& root, MccfrSampler& sampler,
                        const PolicyProfile& compatriots) override final {
    double avg = 0.0;
    for (int player = 0; player < root.NumPlayers(); ++player) {
      auto [v, initial_info_states, cf_value_tree, _] =
          evaluators_[player].ComputeCfValueTreeEvaluation(
              root, compatriots.WithSubstitute(Strategy(player), player), sampler);

      // 更新策略
      learners_[player]->Update(initial_info_states, *cf_value_tree, epsilon_);
      // 计算所有玩家的平均值，该式为平均值公式，等价于 avg = (avg*player + v)/(player+1)
      avg += (v - avg) / (player + 1.0);
    }
    ++iteration_;  // 迭代次数+1
    return avg;
  }

  // 获取最大phi遗憾值
  // std::tuple<double, double, double, double> ReachedPhiRegret(
  //     DecisionPoint& root, MccfrSampler& sampler) override final {
  //   double max_full_phi_regret_fus = 0.0;
  //   double max_full_phi_regret = 0.0;
  //   double max_phi_regret = 0.0;
  //   double sum_phi_regret = 0.0;

  //   for (int player = 0; player < root.NumPlayers(); ++player) {
  //     auto [v, initial_info_states, cf_value_tree, _] =
  //         evaluators_[player].ComputeCfValueTreeEvaluation(root, PolicyRefProfile(learners_),
  //                                                          sampler);  // 使用平均策略计算cfv树
  //     // debug 打印cfv树
  //     // std::cout<< "player " << player << " cfv tree: " << std::endl;
  //     // for (const auto& [iss, node] : *cf_value_tree) {
  //     //   // 打印信息状态字符串
  //     //   std::cout << "InfoState: " << iss << std::endl;
  //     //   // 打印重要性加权到达概率
  //     //   std::cout << "IWRP: " << node.iwrp_ << std::endl;
  //     //   // 打印cfv值
  //     //   std::cout << "CfValues: ";
  //     //   for (size_t aidx = 0; aidx < node.cf_values_.Size(); ++aidx) {
  //     //     std::cout << node.cf_values_[aidx] << " ";
  //     //   }
  //     //   std::cout << std::endl;
  //     // }
  //     auto [player_max_full_phi_regret_fus, player_max_full_phi_regret, player_max_phi_regret,
  //           player_sum_phi_regret] =
  //         learners_[player]->LearnerReachedPhiRegret(initial_info_states, *cf_value_tree);
  //     if (player_max_phi_regret > max_phi_regret) {
  //       max_phi_regret = player_max_phi_regret;
  //     }
  //     sum_phi_regret += player_sum_phi_regret;
  //     if (player_max_full_phi_regret > max_full_phi_regret) {
  //       max_full_phi_regret = player_max_full_phi_regret;
  //     }
  //   }
  //   return {max_full_phi_regret_fus, max_full_phi_regret, max_phi_regret, sum_phi_regret};
  // }

  // 获取所有玩家的phiRegrets
  std::tuple<double, double, double, double> PhiRegrets() const override final {
    double max_full_phi_regret_fus = 0.0;
    double max_full_phi_regret = 0.0;
    double max_phi_regret = 0.0;
    double sum_phi_regret = 0.0;

    for (int player = 0; player < learners_.size(); ++player) {
      auto [player_max_full_phi_regret_fus, player_max_full_phi_regret, player_max_phi_regret,
            player_sum_phi_regret] = learners_[player]->PhiRegrets();
      if (player_max_full_phi_regret_fus > max_full_phi_regret_fus) {
        max_full_phi_regret_fus = player_max_full_phi_regret_fus;
      }
      if (player_max_full_phi_regret > max_full_phi_regret) {
        max_full_phi_regret = player_max_full_phi_regret;
      }
      if (player_max_phi_regret > max_phi_regret) {
        max_phi_regret = player_max_phi_regret;
      }
      sum_phi_regret += player_sum_phi_regret;
    }
    return {max_full_phi_regret_fus, max_full_phi_regret, max_phi_regret, sum_phi_regret};
  }

  // Changing epsilon changes no cached object: the payoff transformation is
  // constructed from epsilon_ in every learner update.
  void ShrinkEpsilon(double new_epsilon) override final {
    if (!std::isfinite(new_epsilon) || new_epsilon < 0.0) {
      throw std::invalid_argument("epsilon must be finite and non-negative");
    }
    epsilon_ = new_epsilon;
  }

  double Epsilon() const override final { return epsilon_; }

  StageRegretMetrics StageMetrics(double w_numerical_eps) const override final {
    StageRegretMetrics metrics;
    for (size_t player = 0; player < learners_.size(); ++player) {
      auto player_metrics = learners_[player]->StageMetrics(w_numerical_eps);
      for (auto& exposure : player_metrics.exposure_entries) {
        exposure.player = static_cast<int>(player);
      }
      if (player_metrics.has_bottleneck) {
        player_metrics.bottleneck_player = static_cast<int>(player);
      }
      metrics.Merge(player_metrics);
    }
    metrics.Finalize();
    return metrics;
  }

  void ResetStageEvaluation() override final {
    for (auto& learner : learners_) learner->ResetStageEvaluation();
  }

  void ResetSolverRegrets() override final {
    for (auto& learner : learners_) learner->ResetSolverRegrets();
  }

  void SetCfrPlus(bool enabled) override final {
    for (auto& learner : learners_) learner->SetCfrPlus(enabled);
  }

  double SolverRegretL1() const override final {
    double total = 0.0;
    for (const auto& learner : learners_) total += learner->SolverRegretL1();
    return total;
  }

  double StageOriginalRegretL1() const override final {
    double total = 0.0;
    for (const auto& learner : learners_) total += learner->StageOriginalRegretL1();
    return total;
  }

  double StageExposureSum() const override final {
    double total = 0.0;
    for (const auto& learner : learners_) total += learner->StageExposureSum();
    return total;
  }

  Cfv UpdateAlternateAndReturnEv(DecisionPoint& root, MccfrSampler& sampler) override final {
    // MapPolicy policy=ToMapPolicy(root);
    // const auto& rt_value_map = ComputePerturbedValue(root, policy);
    double avg = 0.0;

    for (int player = 0; player < root.NumPlayers(); ++player) {
      auto [v, initial_info_states, cf_value_tree, _] =
          evaluators_[player].ComputeCfValueTreeEvaluation(root, PolicyRefProfile(learners_),
                                                           sampler);
      // 计算pcfv
      // const auto rt_cf_value_tree=ComputePerturbedcfValueTree(std::move(*cf_value_tree));
      learners_[player]->Update(initial_info_states, *cf_value_tree, epsilon_);
      avg += (v - avg) / (player + 1.0);
    }
    ++iteration_;

    return avg;
  }

  PolicyProfile Frozen() const override final { return Clone(learners_); }

  // 计算perturbed cfv树，返回一个InfoStateUvm<CfValueTreeNode>指针
  // InfoStateUvm<CfValueTreeNode> ComputePerturbedcfValueTree(InfoStateUvm<CfValueTreeNode>
  // cf_value_tree) const {
  //   auto p_cf_value_tree = std::move(cf_value_tree); // move instead of copy
  //   for (auto& [iss, node] : p_cf_value_tree) {
  //     // expected perturbed value
  //     // 还可以更加general： epsilon 可以使用数组存储对应不同的动作扰动值，但目前没有必要
  //     double p_value = 0.0;
  //     for (size_t aidx = 0; aidx < node.cf_values_.Size(); ++aidx) {
  //       p_value += node.cf_values_[aidx]*epsilon_;
  //     }
  //     for (size_t aidx = 0; aidx < node.cf_values_.Size(); ++aidx) {
  //       node.cf_values_[aidx] = (1-node.cf_values_.Size()*epsilon_)*node.cf_values_[aidx] +
  //       p_value;
  //     }
  //   }
  //   return p_cf_value_tree;
  // }

 private:
  // 创建每个玩家对应的PolicyCfValueTreeEvaluator，将记录对应玩家的cfv树
  std::vector<PolicyCfValueTreeEvaluator> NewEvaluators() {
    std::vector<PolicyCfValueTreeEvaluator> l;
    for (size_t i = 0; i < learners_.size(); ++i) {
      l.emplace_back(i);  // = l.push_back(PolicyCfValueTreeEvaluator(i));
    }
    return l;
  }

 private:
  std::vector<CfValueTreeLearnerPtr> learners_;
  std::vector<PolicyCfValueTreeEvaluator> evaluators_;
  double epsilon_;
  // std::unique_ptr<PolicyProfile> rt_policy_profile_; // rt policy, update for each T iterations
  size_t iteration_;
};

// QBCE算法配置，继承至AdaptiveProfile
class QBCECfTreeLearnerProfile : public AdaptiveProfile {
 public:
  // learners_: 使用智能指针创建一个包含学习者的向量，包含每个玩家 CfValueTreeLearnerPtr
  // evaluators_：每个玩家对应一个PolicyCfValueTreeEvaluator，用于计算该玩家的cfv树,
  // 给定根节点、玩家策略和采样器
  QBCECfTreeLearnerProfile(std::vector<CfValueTreeLearnerPtr>&& learners)
      : learners_(std::move(learners)), evaluators_(NewEvaluators()) {}

  Policy* Strategy(size_t player) const override final { return learners_[player].get(); }
  // 更新策略并返回期望值
  // compatriots: 玩家的策略组，替换player位置的策略为当前玩家的策略
  Cfv UpdateAndReturnEv(DecisionPoint& root, MccfrSampler& sampler,
                        const PolicyProfile& compatriots) override final {
    double avg = 0.0;
    // 对于当前玩家处于不同顺位，计算其cfv树,用于后续更新策略
    for (int player = 0; player < root.NumPlayers(); ++player) {
      // 获取需要更新的初始节点siblings的iss和cfv树
      const auto [v, initial_info_states, cf_value_tree, _] =
          evaluators_[player].ComputeCfValueTreeEvaluation(
              root, compatriots.WithSubstitute(Strategy(player), player), sampler);
      // 使用初始节点的iss和cfv树更新该玩家的策略,使用了入栈和出栈机制
      learners_[player]->Update(initial_info_states, *cf_value_tree);
      // 计算所有玩家的平均值，该式为平均值公式
      // avg=avg+(v-avg)/(player+1)等价于avg=(avg*player+v)/(player+1)，
      // 前者数值稳定性更好，避免了avg*player可能过大导致的溢出问题
      avg += (v - avg) / (player + 1.0);
    }
    return avg;
  }
  // 轮流更新并返回期望值
  Cfv UpdateAlternateAndReturnEv(DecisionPoint& root, MccfrSampler& sampler) override final {
    double avg = 0.0;
    for (int player = 0; player < root.NumPlayers(); ++player) {
      const auto [v, initial_info_states, cf_value_tree, _] =
          evaluators_[player].ComputeCfValueTreeEvaluation(root, PolicyRefProfile(learners_),
                                                           sampler);
      learners_[player]->Update(initial_info_states, *cf_value_tree);
      avg += (v - avg) / (player + 1.0);
    }
    return avg;
  }
  // 函数的签名返回类型，等价return PolicyProfile(Clone(learners_))，隐式调用其构造函数
  PolicyProfile Frozen() const override final { return Clone(learners_); }

  // Aggregate the same cumulative phi-regret metrics used by PBCE.
  std::tuple<double, double, double, double> PhiRegrets() const override final {
    double max_full_phi_regret_fus = 0.0;
    double max_full_phi_regret = 0.0;
    double max_phi_regret = 0.0;
    double sum_phi_regret = 0.0;
    for (const auto& learner : learners_) {
      const auto [full_fus, full, conditional, sum] = learner->PhiRegrets();
      max_full_phi_regret_fus = std::max(max_full_phi_regret_fus, full_fus);
      max_full_phi_regret = std::max(max_full_phi_regret, full);
      max_phi_regret = std::max(max_phi_regret, conditional);
      sum_phi_regret += sum;
    }
    return {max_full_phi_regret_fus, max_full_phi_regret, max_phi_regret, sum_phi_regret};
  }

  double StageOriginalLocalRegret() const override final {
    double maximum = 0.0;
    for (const auto& learner : learners_) {
      const double value = learner->StageOriginalLocalRegret();
      if (!std::isfinite(value)) return value;
      maximum = std::max(maximum, value);
    }
    return maximum;
  }

  void SetCfrPlus(bool enabled) override final {
    for (auto& learner : learners_) learner->SetCfrPlus(enabled);
  }

 private:
  // 创建每个玩家对应的PolicyCfValueTreeEvaluator，将记录对应玩家的cfv树
  std::vector<PolicyCfValueTreeEvaluator> NewEvaluators() {
    std::vector<PolicyCfValueTreeEvaluator> l;
    for (size_t i = 0; i < learners_.size(); ++i) {
      l.emplace_back(i);  // = l.push_back(PolicyCfValueTreeEvaluator(i));
    }
    return l;
  }

 private:
  std::vector<CfValueTreeLearnerPtr> learners_;
  std::vector<PolicyCfValueTreeEvaluator> evaluators_;
  // size_t k_queries_;
  // bool sequential_;
  // bool causal_;
};

// 继承AdaptiveProfile
class MapPolicyAdaptiveProfile : public AdaptiveProfile {
 public:
  MapPolicyAdaptiveProfile(size_t num_players) : AdaptiveProfile(), profile_() {
    profile_.reserve(num_players);
    for (size_t i = 0; i < num_players; ++i) {
      profile_.emplace_back(new MapPolicy());
    }
  }

  Policy* Strategy(size_t player) const override final { return profile_[player].get(); }
  PolicyProfile Frozen() const override final { return Clone(profile_); }

 protected:
  std::vector<MapPolicyPtr> profile_;
};

// 继承MapPolicyAdaptiveProfile
class PolicyIterationProfile : public MapPolicyAdaptiveProfile {
 public:
  PolicyIterationProfile(size_t num_players) : MapPolicyAdaptiveProfile(num_players) {}

  Cfv UpdateAndReturnEv(DecisionPoint& root, MccfrSampler& sampler,
                        const PolicyProfile& compatriots) override final {
    double avg = 0.0;
    for (int player = 0; player < root.NumPlayers(); ++player) {
      const auto [v, _] =
          PolicyValue(root, player, compatriots.WithSubstitute(Strategy(player), player), sampler);
      profile_[player].reset(
          new MapPolicy(BestResponse(compatriots).Policy(root, -player - 1).first));
      avg += (v - avg) / (player + 1.0);
    }
    return avg;
  }

  Cfv UpdateAlternateAndReturnEv(DecisionPoint& root, MccfrSampler& sampler) override final {
    double avg = 0.0;
    for (int player = 0; player < root.NumPlayers(); ++player) {
      const auto [v, _] = PolicyValue(root, player, PolicyRefProfile(profile_), sampler);
      profile_[player].reset(
          new MapPolicy(BestResponse(PolicyRefProfile(profile_)).Policy(root, -player - 1).first));
      avg += (v - avg) / (player + 1.0);
    }
    return avg;
  }
};

class AntiPolicyIterationProfile : public MapPolicyAdaptiveProfile {
 public:
  AntiPolicyIterationProfile(size_t num_players)
      : MapPolicyAdaptiveProfile(num_players), policy_iteration_(num_players) {}

  Cfv UpdateAndReturnEv(DecisionPoint& root, MccfrSampler& sampler,
                        const PolicyProfile& compatriots) override final {
    double avg = 0.0;
    for (int player = 0; player < root.NumPlayers(); ++player) {
      const auto [v, _] =
          PolicyValue(root, player, compatriots.WithSubstitute(Strategy(player), player), sampler);
      avg += (v - avg) / (player + 1.0);
    }
    policy_iteration_.UpdateAndReturnEv(root, sampler, compatriots);
    const auto pi_profile = policy_iteration_.Frozen();

    for (int player = 0; player < root.NumPlayers(); ++player) {
      profile_[player].reset(
          new MapPolicy(BestResponse(pi_profile).Policy(root, -player - 1).first));
    }
    return avg;
  }

  Cfv UpdateAlternateAndReturnEv(DecisionPoint& root, MccfrSampler& sampler) override final {
    double avg = 0.0;
    for (int player = 0; player < root.NumPlayers(); ++player) {
      const auto [v, _] = PolicyValue(root, player, PolicyRefProfile(profile_), sampler);
      avg += (v - avg) / (player + 1.0);

      policy_iteration_.UpdateAndReturnEv(root, sampler, Frozen());
      profile_[player].reset(
          new MapPolicy(BestResponse(policy_iteration_.Frozen()).Policy(root, -player - 1).first));
    }
    return avg;
  }

 private:
  PolicyIterationProfile policy_iteration_;
};

class BestResponseProfile : public MapPolicyAdaptiveProfile {
 public:
  BestResponseProfile(size_t num_players) : MapPolicyAdaptiveProfile(num_players) {}

  Cfv UpdateAndReturnEv(DecisionPoint& root, MccfrSampler& sampler,
                        const PolicyProfile& compatriots) override final {
    double avg = 0.0;
    for (int player = 0; player < root.NumPlayers(); ++player) {
      auto [policy, v] = BestResponse(compatriots).Policy(root, -player - 1);
      profile_[player].reset(new MapPolicy(std::move(policy)));
      avg += (v - avg) / (player + 1.0);
    }
    return avg;
  }

  Cfv UpdateAlternateAndReturnEv(DecisionPoint& root, MccfrSampler& sampler) override final {
    double avg = 0.0;
    for (int player = 0; player < root.NumPlayers(); ++player) {
      auto [policy, v] = BestResponse(PolicyRefProfile(profile_)).Policy(root, -player - 1);
      profile_[player].reset(new MapPolicy(std::move(policy)));
      avg += (v - avg) / (player + 1.0);
    }
    return avg;
  }
};

class FictitiousPlayProfile : public MapPolicyAdaptiveProfile {
 public:
  FictitiousPlayProfile(size_t num_players)
      : MapPolicyAdaptiveProfile(num_players), compatriot_empirical_play_() {}

  Cfv UpdateAndReturnEv(DecisionPoint& root, MccfrSampler& sampler,
                        const PolicyProfile& compatriots) override final {
    double avg = 0.0;
    for (int player = 0; player < root.NumPlayers(); ++player) {
      const auto [v, _] =
          PolicyValue(root, player, compatriots.WithSubstitute(Strategy(player), player), sampler);
      avg += (v - avg) / (player + 1.0);
    }
    compatriot_empirical_play_.Avg(compatriots, root);

    for (int player = 0; player < root.NumPlayers(); ++player) {
      profile_[player].reset(
          new MapPolicy(BestResponse(compatriot_empirical_play_).Policy(root, -player - 1).first));
    }
    return avg;
  }

  Cfv UpdateAlternateAndReturnEv(DecisionPoint& root, MccfrSampler& sampler) override final {
    double avg = 0.0;
    for (int player = 0; player < root.NumPlayers(); ++player) {
      const auto profile_ref = PolicyRefProfile(profile_);
      const auto [v, _] = PolicyValue(root, player, profile_ref, sampler);
      avg += (v - avg) / (player + 1.0);

      const int not_player = -player - 1;
      compatriot_empirical_play_.Avg(profile_ref, root, not_player);
      profile_[player].reset(
          new MapPolicy(BestResponse(compatriot_empirical_play_).Policy(root, not_player).first));
    }
    return avg;
  }

 private:
  MapPolicy compatriot_empirical_play_;
};
}  // namespace hr_edl

#endif  // HR_EDL_ADAPTIVE_PROFILE_H_
