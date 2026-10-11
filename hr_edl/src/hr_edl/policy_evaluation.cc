#include "hr_edl/policy_evaluation.h"

#include <cassert>

using namespace open_spiel;

namespace hr_edl {

Cfv PolicyRegretAndReachProbEvaluator::operator()(
    const open_spiel::State& state, double chance_reach_iw,
    double player_sampling_prob) {
  if (state.IsTerminal()) {
    return state.Returns()[regret_player_] * chance_reach_iw *
           CounterfactualReachProb() / player_sampling_prob;
  } else if (state.IsChanceNode()) {
    Cfv state_value = 0.0;
    sampler_.SampleChanceOutcomes(
        state,
        [this, &state, &state_value, chance_reach_iw, player_sampling_prob](
            const ActionAndProb& outcome, double importance_weight) {
          state_value += (*this)(*state.Child(outcome.first),
                                 chance_reach_iw * importance_weight,
                                 player_sampling_prob);
        });
    return state_value;
  }

  ++num_decision_histories_;

  const int current_player = state.CurrentPlayer();
  const std::vector<open_spiel::Action> legal_actions =
      state.LegalActions(current_player);
  const std::vector<double>& policy = profile_.Response(state);
  const std::string info_state = state.InformationStateString(current_player);
  const int num_legal_actions = legal_actions.size();
  const double my_reach_prob = reach_probabilities_[current_player];

  if (SaveReachProbs(current_player)) {
    auto& reach_probs = GetOrCreate<std::string, std::vector<double>>(
        reach_probs_table_, info_state, [num_legal_actions]() {
          return std::vector<double>(num_legal_actions, 0.0);
        });
    for (int action_idx = 0; action_idx < num_legal_actions; ++action_idx) {
      const auto child_prob = policy[action_idx];

      if (child_prob > 0) {
        const auto reach_importance_weight =
            my_reach_prob / player_sampling_prob;
        reach_probs[action_idx] += reach_importance_weight * child_prob;
      }
    }
  }

  Cfv state_value = 0.0;
  if (SaveRegrets(current_player)) {
    double action_values[num_legal_actions];
    sampler_.SampleTargetPlayerActions(
        policy, [this, &action_values, &legal_actions, &state, chance_reach_iw,
                 player_sampling_prob, &state_value](
                    int action_idx, double action_prob, double sampling_prob) {
          const Cfv child_utility =
              (*this)(*state.Child(legal_actions[action_idx]), chance_reach_iw,
                      player_sampling_prob * sampling_prob);
          action_values[action_idx] = child_utility;
          const Cfv weighted_child_utility = action_prob * child_utility;
          state_value += weighted_child_utility;
        });
    auto& regrets = GetOrCreate<std::string, CfValues>(
        regret_table_, info_state, [num_legal_actions]() -> CfValues {
          return {CfActionValues(num_legal_actions, 0), 0};
        });
    regrets.ev_ += state_value;
    for (size_t a = 0; a < num_legal_actions; ++a) {
      regrets.v_[a] += action_values[a];
    }
  } else {
    sampler_.SampleExternalPlayerActions(
        policy, [this, my_reach_prob, chance_reach_iw, player_sampling_prob,
                 &legal_actions, current_player, &state_value, &state](
                    int action_idx, double action_prob, double sampling_prob) {
          const auto child_reach_probability = my_reach_prob * action_prob;

          if (!SaveReachProbs(current_player) || child_reach_probability > 0) {
            reach_probabilities_[current_player] = child_reach_probability;

            state_value +=
                (*this)(*state.Child(legal_actions[action_idx]),
                        chance_reach_iw, player_sampling_prob * sampling_prob);
          }
        });

    reach_probabilities_[current_player] = my_reach_prob;
  }
  return state_value;
}

Cfv PolicyRegretAndReachProbEvaluator::CounterfactualValue(
    DecisionPoint& decision_point, double chance_reach_iw,
    double player_sampling_prob, int action_idx) {
  Cfv v = 0;
  sampler_.SampleChanceOutcomes(
      decision_point.OutcomeProbabilities(action_idx),
      [this, action_idx, &decision_point, chance_reach_iw, player_sampling_prob,
       &v](int outcome, double outcome_prob, double outcome_sampling_prob) {
        decision_point.Apply(action_idx, outcome);
        v += (*this)(decision_point,
                     chance_reach_iw * outcome_prob / outcome_sampling_prob,
                     player_sampling_prob);
        decision_point.Undo();
      });
  return v;
}

Cfv PolicyRegretAndReachProbEvaluator::CounterfactualValue(
    DecisionPoint& decision_point, double chance_reach_iw,
    double player_sampling_prob, int action_idx, double next_reach_prob) {
  Cfv v = 0;
  sampler_.SampleChanceOutcomes(
      decision_point.OutcomeProbabilities(action_idx),
      [this, action_idx, &decision_point, chance_reach_iw, player_sampling_prob,
       next_reach_prob,
       &v](int outcome, double outcome_prob, double outcome_sampling_prob) {
        reach_probabilities_[decision_point.PlayerToAct()] = next_reach_prob;
        decision_point.Apply(action_idx, outcome);
        v += (*this)(decision_point,
                     chance_reach_iw * outcome_prob / outcome_sampling_prob,
                     player_sampling_prob);
        decision_point.Undo();
      });
  return v;
}

Cfv PolicyRegretAndReachProbEvaluator::operator()(DecisionPoint& decision_point,
                                                  double chance_reach_iw,
                                                  double player_sampling_prob) {
  if (decision_point.IsTerminal()) {
    return Return(decision_point) * chance_reach_iw *
           CounterfactualReachProb() / player_sampling_prob;
  }
  ++num_decision_histories_;

  const int current_player = decision_point.PlayerToAct();
  assert(decision_point.OpenSpielStatePtr());
  const std::vector<double> policy =
      profile_.Response(*decision_point.OpenSpielStatePtr());
  const std::string info_state = decision_point.InformationStateString();
  const int num_legal_actions = decision_point.NumActions();
  const double my_reach_prob = reach_probabilities_[current_player];

  if (SaveReachProbs(current_player)) {
    auto& reach_probs = GetOrCreate<std::string, std::vector<double>>(
        reach_probs_table_, info_state, [num_legal_actions]() {
          return std::vector<double>(num_legal_actions, 0.0);
        });
    for (int action_idx = 0; action_idx < num_legal_actions; ++action_idx) {
      reach_probs[action_idx] +=
          policy[action_idx] * my_reach_prob / player_sampling_prob;
    }
  }

  Cfv state_value = 0.0;
  if (SaveRegrets(current_player)) {
    double action_values[num_legal_actions];
    sampler_.SampleTargetPlayerActions(
        policy, [this, &decision_point, chance_reach_iw, player_sampling_prob,
                 &action_values, &state_value](
                    int action_idx, double action_prob, double sampling_prob) {
          const Cfv cfv = this->CounterfactualValue(
              decision_point, chance_reach_iw,
              player_sampling_prob * sampling_prob, action_idx);
          action_values[action_idx] = cfv;
          state_value += action_prob * cfv;
        });
    auto& regrets = GetOrCreate<std::string, CfValues>(
        regret_table_, info_state, [num_legal_actions]() -> CfValues {
          return {CfActionValues(num_legal_actions, 0), 0};
        });
    regrets.ev_ += state_value;
    for (size_t a = 0; a < num_legal_actions; ++a) {
      regrets.v_[a] += action_values[a];
    }
  } else {
    sampler_.SampleExternalPlayerActions(
        policy, [this, my_reach_prob, chance_reach_iw, player_sampling_prob,
                 current_player, &state_value, &decision_point](
                    int action_idx, double action_prob, double sampling_prob) {
          const double next_reach_probability = my_reach_prob * action_prob;
          if (!SaveReachProbs(current_player) || next_reach_probability > 0) {
            state_value +=
                CounterfactualValue(decision_point, chance_reach_iw,
                                    player_sampling_prob * sampling_prob,
                                    action_idx, next_reach_probability);
          }
        });
    reach_probabilities_[current_player] = my_reach_prob;
  }
  return state_value;
}

std::tuple<Cfv,
           std::pair<InfoStateUvm<CfValues>, InfoStateUvm<std::vector<double>>>,
           int>
PolicyRegretsAndReachProbs(const open_spiel::State& root, int regret_player,
                           const Policy& profile, MccfrSampler& sampler,
                           int reach_prob_player) {
  PolicyRegretAndReachProbEvaluator evaluator(
      regret_player, profile, sampler, root.NumPlayers(), reach_prob_player);
  const Cfv ev = evaluator(root, 1, 1);
  return {ev,
          {std::move(evaluator.regret_table_),
           std::move(evaluator.reach_probs_table_)},
          evaluator.num_decision_histories_};
}

std::tuple<Cfv,
           std::pair<InfoStateUvm<CfValues>, InfoStateUvm<std::vector<double>>>,
           int>
PolicyRegretsAndReachProbs(DecisionPoint& root, int regret_player,
                           const Policy& profile, MccfrSampler& sampler,
                           int reach_prob_player) {
  PolicyRegretAndReachProbEvaluator evaluator(
      regret_player, profile, sampler, root.NumPlayers(), reach_prob_player);
  const Cfv ev = evaluator.CounterfactualValue(root, 1.0, 1.0, 0);
  return {ev,
          {std::move(evaluator.regret_table_),
           std::move(evaluator.reach_probs_table_)},
          evaluator.num_decision_histories_};
}

Cfv PolicyRegretEvaluator::CounterfactualValue(
    DecisionPoint& decision_point, double importance_weight, int action_idx,
    const std::function<Cfv(DecisionPoint&, double)>& backup) {
  Cfv v = 0;
  sampler_.SampleChanceOutcomes(
      decision_point.OutcomeProbabilities(action_idx),
      [action_idx, &decision_point, importance_weight, &backup, &v](
          int outcome, double outcome_prob, double outcome_sampling_prob) {
        decision_point.Apply(action_idx, outcome);
        v += backup(decision_point,
                    importance_weight * outcome_prob / outcome_sampling_prob);
        decision_point.Undo();
      });
  return v;
}

Cfv PolicyRegretEvaluator::CounterfactualValue(
    DecisionPoint& decision_point, double importance_weight, int action_idx,
    double next_reach_prob,
    const std::function<Cfv(DecisionPoint&, double)>& backup) {
  Cfv v = 0;
  sampler_.SampleChanceOutcomes(
      decision_point.OutcomeProbabilities(action_idx),
      [this, action_idx, &decision_point, importance_weight, next_reach_prob,
       &backup,
       &v](int outcome, double outcome_prob, double outcome_sampling_prob) {
        reach_probabilities_[decision_point.PlayerToAct()] = next_reach_prob;
        decision_point.Apply(action_idx, outcome);
        v += backup(decision_point,
                    importance_weight * outcome_prob / outcome_sampling_prob);
        decision_point.Undo();
      });
  return v;
}

Cfv PolicyRegretEvaluator::ComputeCounterfactualRegrets(
    DecisionPoint& decision_point, double importance_weight) {
  if (decision_point.IsTerminal()) {
    return Return(decision_point) * importance_weight *
           CounterfactualReachProb();
  }
  ++num_decision_histories_;

  assert(decision_point.OpenSpielStatePtr());
  const std::vector<double> policy =
      profile_.Response(*decision_point.OpenSpielStatePtr());

  Cfv state_value = 0.0;
  if (SaveRegrets(decision_point.PlayerToAct())) {
    double action_values[decision_point.NumActions()];
    sampler_.SampleTargetPlayerActions(
        policy, [this, &decision_point, importance_weight, &action_values,
                 &state_value](int action_idx, double action_prob,
                               double sampling_prob) {
          const Cfv cfv = this->CounterfactualValue(
              decision_point, importance_weight / sampling_prob, action_idx,
              [this](DecisionPoint& next_dp, double next_iw) {
                return this->ComputeCounterfactualRegrets(next_dp, next_iw);
              });
          action_values[action_idx] = cfv;
          state_value += action_prob * cfv;
        });
    const std::string info_state = decision_point.InformationStateString();
    const int num_legal_actions = decision_point.NumActions();
    auto& regrets = GetOrCreate<std::string, CfValues>(
        regret_table_, info_state, [num_legal_actions]() -> CfValues {
          return {CfActionValues(num_legal_actions, 0), 0};
        });
    regrets.ev_ += state_value;
    for (size_t a = 0; a < num_legal_actions; ++a) {
      regrets.v_[a] += action_values[a];
    }
  } else {
    const double my_reach_prob =
        reach_probabilities_[decision_point.PlayerToAct()];
    sampler_.SampleExternalPlayerActions(
        policy,
        [this, my_reach_prob, importance_weight, &state_value, &decision_point](
            int action_idx, double action_prob, double sampling_prob) {
          const double next_reach_probability = my_reach_prob * action_prob;
          if (next_reach_probability > 0) {
            state_value += CounterfactualValue(
                decision_point, importance_weight / sampling_prob, action_idx,
                next_reach_probability,
                [this](DecisionPoint& next_dp, double next_iw) {
                  return this->ComputeCounterfactualRegrets(next_dp, next_iw);
                });
          }
        });
    reach_probabilities_[decision_point.PlayerToAct()] = my_reach_prob;
  }
  return state_value;
}

Cfv PolicyRegretEvaluator::ComputeReachWeightedRegrets(
    DecisionPoint& decision_point, double importance_weight) {
  if (decision_point.IsTerminal()) {
    return Return(decision_point) * importance_weight *
           CounterfactualReachProb();
  }
  ++num_decision_histories_;

  const int current_player = decision_point.PlayerToAct();
  assert(decision_point.OpenSpielStatePtr());
  const std::vector<double> policy =
      profile_.Response(*decision_point.OpenSpielStatePtr());
  const double my_reach_prob = reach_probabilities_[current_player];

  Cfv state_value = 0.0;
  if (SaveRegrets(current_player)) {
    double action_values[decision_point.NumActions()];
    sampler_.SampleTargetPlayerActions(
        policy, [this, &decision_point, importance_weight, my_reach_prob,
                 &action_values, &state_value](
                    int action_idx, double action_prob, double sampling_prob) {
          const Cfv cfv = this->CounterfactualValue(
              decision_point, importance_weight / sampling_prob, action_idx,
              my_reach_prob * action_prob,
              [this](DecisionPoint& next_dp, double next_iw) {
                return this->ComputeReachWeightedRegrets(next_dp, next_iw);
              });
          action_values[action_idx] = my_reach_prob * cfv;
          state_value += action_prob * cfv;
        });
    const std::string info_state = decision_point.InformationStateString();
    const size_t n = decision_point.NumActions();
    auto& regrets = GetOrCreate<std::string, CfValues>(
        regret_table_, info_state, [n]() -> CfValues {
          return {CfActionValues(n, 0), 0};
        });
    regrets.ev_ += my_reach_prob * state_value;
    for (size_t a = 0; a < n; ++a) {
      regrets.v_[a] += action_values[a];
    }
  } else {
    sampler_.SampleExternalPlayerActions(
        policy, [this, my_reach_prob, importance_weight, current_player,
                 &state_value, &decision_point](
                    int action_idx, double action_prob, double sampling_prob) {
          const double next_reach_probability = my_reach_prob * action_prob;
          if (next_reach_probability > 0) {
            state_value += CounterfactualValue(
                decision_point, importance_weight / sampling_prob, action_idx,
                next_reach_probability,
                [this](DecisionPoint& next_dp, double next_iw) {
                  return this->ComputeReachWeightedRegrets(next_dp, next_iw);
                });
          }
        });
  }
  reach_probabilities_[current_player] = my_reach_prob;
  return state_value;
}

std::tuple<Cfv, InfoStateUvm<CfValues>, int> PolicyCounterfactualRegrets(
    DecisionPoint& root, int regret_player, const Policy& profile,
    MccfrSampler& sampler) {
  PolicyRegretEvaluator evaluator(regret_player, profile, sampler,
                                  root.NumPlayers());
  const Cfv ev = evaluator.CounterfactualValue(
      root, 1.0, 0, [&evaluator](DecisionPoint& next_dp, double next_iw) {
        return evaluator.ComputeCounterfactualRegrets(next_dp, next_iw);
      });
  return {ev, std::move(evaluator.regret_table_),
          evaluator.num_decision_histories_};
}

std::tuple<Cfv, InfoStateUvm<CfValues>, int> PolicyReachWeightedRegrets(
    DecisionPoint& root, int regret_player, const Policy& profile,
    MccfrSampler& sampler) {
  PolicyRegretEvaluator evaluator(regret_player, profile, sampler,
                                  root.NumPlayers());
  const Cfv ev = evaluator.CounterfactualValue(
      root, 1.0, 0, [&evaluator](DecisionPoint& next_dp, double next_iw) {
        return evaluator.ComputeReachWeightedRegrets(next_dp, next_iw);
      });
  return {ev, std::move(evaluator.regret_table_),
          evaluator.num_decision_histories_};
}

Cfv PolicyValueEvaluator::operator()(DecisionPoint& decision_point,
                                     double importance_weighted_reach_prob,
                                     int action_idx) {
  Cfv v = 0;
  sampler_.SampleChanceOutcomes(
      decision_point.OutcomeProbabilities(action_idx),
      [this, action_idx, &decision_point, importance_weighted_reach_prob, &v](
          int outcome, double outcome_prob, double outcome_sampling_prob) {
        decision_point.Apply(action_idx, outcome);
        v += (*this)(decision_point, importance_weighted_reach_prob *
                                         outcome_prob / outcome_sampling_prob);
        decision_point.Undo();
      });
  return v;
}

Cfv PolicyValueEvaluator::operator()(DecisionPoint& decision_point,
                                     double importance_weighted_reach_prob) {
  if (decision_point.IsTerminal()) {
    return Return(decision_point) * importance_weighted_reach_prob;
  }
  ++num_decision_histories_;

  assert(decision_point.OpenSpielStatePtr());
  const std::vector<double> policy =
      profile_.Response(*decision_point.OpenSpielStatePtr());

  Cfv state_value = 0;
  if (decision_point.PlayerToAct() == player_) {
    sampler_.SampleTargetPlayerActions(
        policy,
        [this, &decision_point, importance_weighted_reach_prob, &state_value](
            int action_idx, double action_prob, double sampling_prob) {
          if (action_prob > 0) {
            state_value += (*this)(
                decision_point,
                action_prob * importance_weighted_reach_prob / sampling_prob,
                action_idx);
          }
        });
  } else {
    sampler_.SampleExternalPlayerActions(
        policy,
        [this, importance_weighted_reach_prob, &state_value, &decision_point](
            int action_idx, double action_prob, double sampling_prob) {
          if (action_prob > 0) {
            state_value += (*this)(
                decision_point,
                action_prob * importance_weighted_reach_prob / sampling_prob,
                action_idx);
          }
        });
  }
  return state_value;
}

// 计算player策略在决策点root的期望价值和决策节点数量，给定策略profile和采样器sampler
std::pair<Cfv, int> PolicyValue(DecisionPoint& root, int player,
                                const Policy& profile, MccfrSampler& sampler) {
  PolicyValueEvaluator evaluator(player, profile, sampler, root.NumPlayers()); // 实例化PolicyValueEvaluator类，用来计算策略的价值
  const Cfv ev = evaluator(root, 1.0, 0); //使用了重载运算符operate()，计算决策点的期望价值
  return {ev, evaluator.num_decision_histories_};
}

// 计算cfv tree的期望价值,
// 注意计算到达概率
Cfv PolicyCfValueTreeEvaluator::ComputeCfValueTree(
    std::vector<std::string>& siblings, DecisionPoint& decision_point,
    const Policy& profile, MccfrSampler& sampler,
    double importance_weighted_reach_prob) {
  ++num_decision_histories_; // 决策节点数量加1

  assert(decision_point.OpenSpielStatePtr());
    // 获取当前状态的策略, 根据is_current_strategy选择是当前策略还是平均策略
  std::vector<double> policy;
  policy = profile.Response(*decision_point.OpenSpielStatePtr());

  // 当前决策点的期望价值，指向目标玩家；当该节点是外部玩家的决策节点时，计算后续节点的期望值并乘以外部玩家的动作概率
  Cfv state_value = 0.0; 
  if (SaveRegrets(decision_point.PlayerToAct())) { // 是否需要保存遗憾值->当前决策点是行动玩家
    // 初始化cfv tree节点
    // 如果cf_value_tree_中不包含当前决策点的信息状态字符串，创建一个新的CfValueTreeNode
    if (!cf_value_tree_.contains(decision_point.InformationStateStringRef())) { 
      cf_value_tree_.emplace(decision_point.InformationStateStringRef(), // emplace：在容器中指定位置构造元素,如果是vector，指定idx，如果是map,指定key
                             decision_point.NumActions()); // key: 决策点的状态字符串，value: 构造CfValueTreeNode(num_actions)
      siblings.push_back(decision_point.InformationStateStringRef()); // 将目标玩家决策点的信息状态字符串添加到siblings中
    }
    //计算cfv值
    double action_values[decision_point.NumActions()]; // 动作价值
    // double iwrp = 0.0; // 重要性加权到达概率
    sampler.SampleTargetPlayerActions( // 遍历采样当前玩家的所有动作
        policy, [this, &decision_point, importance_weighted_reach_prob,
                 &action_values, &state_value, &sampler, &profile](
                    int action_idx, double action_prob, double sampling_prob) {
          std::vector<std::string> child_keys;
          const Cfv cfv = CounterfactualValue(
              child_keys, decision_point, profile, sampler,
              importance_weighted_reach_prob / sampling_prob, action_idx);
          action_values[action_idx] = cfv; // 当前动作的cfv值，子节点的期望值
          state_value += action_prob * cfv; // 期望值=sum(cfv*动作概率)
          // iwrp += action_prob * importance_weighted_reach_prob / sampling_prob;
          if (child_keys.size() > 0) { // 如果当前动作有后继节点
            // 将当前动作的后继节点信息状态字符串添加到cf_value_tree_中
            Concat(cf_value_tree_.at(decision_point.InformationStateStringRef())
                       .child_keys_[action_idx],
                   child_keys);
          }
        });
    auto& cf_values =
        cf_value_tree_.at(decision_point.InformationStateStringRef())
            .cf_values_; //引用当前决策节点的CfValues结构体
    // 更新重要性权重到达概率 perfect recall, Bayes rule 中到达概率是相等的,非完全信息中到达概率可能不同
    cf_value_tree_.at(decision_point.InformationStateStringRef()).iwrp_ += 
        importance_weighted_reach_prob; 
    cf_values.ev_ += state_value; // 更新期望值
    // debug 检查decision_point 是 2b 时的cfv信息
    // if (decision_point.InformationStateString() == "2b") {
    //   SPIEL_CHECK_EQ(state_value, 0.0);
    // }
    assert(decision_point.NumActions() == cf_values.Size());
    // 更新每个动作的cfv值
    for (size_t a = 0; a < decision_point.NumActions(); ++a) {
      cf_values.v_[a] += action_values[a];
    }
  } else { // 如果不需要保存遗憾值
    // 遍历采样所有外部玩家的动作
    sampler.SampleExternalPlayerActions(
        policy, [this, importance_weighted_reach_prob, &state_value,
                 &decision_point, &siblings, &sampler, &profile](
                    int action_idx, double action_prob, double sampling_prob) {
          if (action_prob >= 0) { // >0 只计算有意义的动作
            state_value += CounterfactualValue( // 期望值 += 外部玩家的动作概率*重要性权重概率
                siblings, decision_point, profile, sampler,
                action_prob * importance_weighted_reach_prob / sampling_prob, 
                action_idx);
          }
        });
  }
  return state_value;
}

// 计算决策点特定动作idx的期望价值
// 这里的siblings是一个字符串向量，用来存储具有（同一个父节点的）兄弟节点信息状态字符串
// decision_point是当前决策点，profile是策略，sampler是采样器，importance_weighted_reach_prob是重要性加权的到达概率， action_idx是动作索引
Cfv PolicyCfValueTreeEvaluator::CounterfactualValue(
    std::vector<std::string>& siblings, DecisionPoint& decision_point,
    const Policy& profile, MccfrSampler& sampler,
    double importance_weighted_reach_prob, int action_idx) {
  Cfv v = 0;
  // 采样当前动作的所有可能结果 outcomes
  sampler.SampleChanceOutcomes(
      decision_point.OutcomeProbabilities(action_idx),// 获取当前动作的outcomes的机会概率分布，如果不是机会节点，则返回全1的分布
      [this, action_idx, &decision_point, importance_weighted_reach_prob, &v,
       &siblings, &profile, &sampler](int outcome, double outcome_prob, 
                                      double outcome_sampling_prob) {
        decision_point.Apply(action_idx, outcome); // 执行动作，并缓存后继节点信息，idx指向子节点outcome
        const double next_iwrp = importance_weighted_reach_prob * outcome_prob / // 节点到达机会概率
                                 outcome_sampling_prob; // outcome_sampling_prob：1.0
        v += decision_point.IsTerminal()
                 ? Return(decision_point) * next_iwrp
                 : ComputeCfValueTree(siblings, decision_point, profile,
                                      sampler, next_iwrp);
        decision_point.Undo(); // 只将索引指向父节点，但是缓存信息还在
      });
  return v;
}

}  // namespace hr_edl
