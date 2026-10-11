#include "hr_edl/samplers.h"

#include "open_spiel/spiel_utils.h"

namespace hr_edl {

// 采样一个动作索引，epsilon-贪心策略
// policy: 策略向量，random_number: 随机数[0,1]，epsilon: 探索率[0,1]
// 返回值：采样得到的动作索引
int SampleActionIndex(const std::vector<double>& policy, double random_number, // NOTE: 这个random_number是怎么生成的？
                      double epsilon) { // 头文件中已经制定了epsilon=0
  double cumulative_prob = 0;
  int aidx = 0;
  for (; aidx < policy.size(); ++aidx) {
    const double prob =
        epsilon / policy.size() + (1.0 - epsilon) * policy[aidx];
    cumulative_prob += prob;
    if (cumulative_prob > random_number) {
      return aidx;
    }
  }
  return aidx - 1;
}

// 采样一个动作索引
// actions_and_probs: 动作和概率的向量，random_number: 随机数
// 返回值：采样得到的动作索引
int SampleActionIndex(const open_spiel::ActionsAndProbs& actions_and_probs,
                      double random_number) {
  double cumulative_prob = 0;
  int a = 0;
  for (; a < actions_and_probs.size(); ++a) {
    cumulative_prob += actions_and_probs[a].second;
    if (cumulative_prob > random_number) {
      return a;
    }
  }
  return a - 1;
}

// 采样所有机会节点的结果
void SampleAllChanceOutcomes(
    const open_spiel::State& state,
    const std::function<void(const ActionAndProb&, double)>& f) {
  for (const auto outcome : state.ChanceOutcomes()) { // std::vector<std::pair<Action, double>>
    f(outcome, outcome.second);
  }
}

// 采样一个机会节点的结果
// random_number: 随机数，state: 当前状态，f: 回调函数
// f会被调用一次，传入采样的结果和其概率
void SampleOneChanceOutcome(
    double random_number, const open_spiel::State& state,
    const std::function<void(const ActionAndProb&, double)>& f) {
  const auto outcomes = state.ChanceOutcomes();
  f(outcomes[SampleActionIndex(outcomes, random_number)], 1.0);
}

// 遍历采样目标玩家的所有动作
void SampleAllTargetPlayerActions(
    const std::vector<double>& policy,
    const std::function<void(int action_idx, double policy_prob,
                             double sampling_prob)>& f) {
  for (int action_idx = 0; action_idx < policy.size(); ++action_idx) {
    f(action_idx, policy[action_idx], 1.0);
  }
}

// 采样目标玩家的一个动作 epsilon-贪心采样
void SampleOneTargetPlayerAction(
    double random_number, const std::vector<double>& policy,
    const std::function<void(int action_idx, double policy_prob,
                             double sampling_prob)>& f,
    double epsilon) {
  const int action_idx = SampleActionIndex(policy, random_number, epsilon); // epsilon-贪心采样,获取动作索引
  const double p = policy[action_idx]; // 动作概率
  f(action_idx, p, (1 - epsilon) * p + epsilon / policy.size()); // epsilon in [0,1], 控制随机性
}

// 遍历采样外部玩家的所有动作
// policy: 策略向量，f: 回调函数
void SampleAllExternalPlayerActions(
    const std::vector<double>& policy,
    // policy_prob: 策略概率，sampling_prob: 采样概率
    const std::function<void(int action_idx, double policy_prob,
                             double sampling_prob)>& f) {
  for (int action_idx = 0; action_idx < policy.size(); ++action_idx) {
    f(action_idx, policy[action_idx], 1.0);
  }
}

// 采样外部玩家的一个动作
// random_number: 随机数，policy: 策略向量，f: 回调函数
void SampleOneExternalPlayerAction(
    double random_number, const std::vector<double>& policy,
    const std::function<void(int action_idx, double policy_prob,
                             double sampling_prob)>& f) {
  const int action_idx = SampleActionIndex(policy, random_number); // 没有提供epsilon，直接按策略采样，这样不会报错吗？
  const double p = policy[action_idx];
  f(action_idx, p, p);
}

}  // namespace hr_edl
