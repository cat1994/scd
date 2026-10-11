#include "hr_edl/decision_point.h"

#include "absl/container/flat_hash_set.h"
#include "hr_edl/containers.h"
#include "hr_edl/samplers.h"
namespace hr_edl {

double DecisionPoint::ApplySampledOutcome(size_t action, double random_number) {
  const std::vector<double>& pi = OutcomeProbabilities(action);
  const size_t outcome = SampleActionIndex(pi, random_number);
  const double prob = pi[outcome];
  Apply(action, outcome);
  return prob;
}

// 构造函数（指针-右值引用），接收一个OpenSpiel状态指针
CachedDecisionPoint::CachedDecisionPoint(open_spiel::StatePtr&& root, // 传入open_spiel::StatePtr类型的根节点
                                         bool save_root, bool save_terminals) // false, true
    : DecisionPoint(root->NumPlayers(), root->NumDistinctActions()), // 调用基类构造函数，传入玩家数量和根节点动作数量
      idx_(0),
      histories_(), // 初始化决策节点的历史缓存向量
      save_terminals_(save_terminals) {
  // 如果根节点是终端状态，在histories_中保存终端节点
  if (root->IsTerminal()) {
    if (save_terminals_) {
      histories_.emplace_back(std::move(root), idx_);
    } else {
      histories_.emplace_back(root->Returns(), idx_);
    }
  // 如果根节点是机会节点且需要保存，默认false不执行该过程
  } else if (root->IsChanceNode() && save_root) { // save_root: false
    histories_.emplace_back(std::move(root));
    histories_[idx_].outcomes_.emplace_back();
    RecursiveCache(*(histories_[idx_].os_state_), 1.0, 0);
    // 根节点不是机会节点且需要保存，默认执行该过程
  } else {
    // 将根节点存储在histories_中， HistoryCache构造函数 for Artificial root
    histories_.emplace_back(); // histores_ 初始化一个根节点 Artificial root 0， 通过emplace_back直接调用默认构造函数
    histories_[idx_].outcomes_.emplace_back(); // Artifical root下初始化一个Outcomes
    RecursiveCache(std::move(root), 1.0, 0); // root 是Artifical root的子节点，这个0唯一动作，后续将0动作对应的所有非机会节点outcome缓存进去
  }
}

// 将机会节点放入到缓存中
// child: 决策节点状态引用
void CachedDecisionPoint::RecursiveCache(const open_spiel::State& child,
                                         double prob, size_t aidx) {
  // 添加所有机会节点的后继节点
  for (const auto [outcome, next_prob] : child.ChanceOutcomes()) {
    RecursiveCache(child.Child(outcome), prob * next_prob, aidx); // 合并连续的机会节点，以累乘概率表示
  }
}

// 递归地缓存决策点的非机会子节点
// 如果是机会节点，递归调用找到非机会节点，直到找到叶子节点和一般决策节点后缓存，机会节点实质上通过概率隐式表示
// child: 子节点状态指针
// prob: 当前机会概率
void CachedDecisionPoint::RecursiveCache(open_spiel::StatePtr&& child,
                                         double prob, size_t aidx) {
  // 机会节点，合并连续的机会节点，递归添加后续
  if (child->IsChanceNode()) {
    RecursiveCache(*child, prob, aidx); // 这是指针的引用，更新了机会节点的概率prob
    return;
  } else if (!child->IsTerminal()) { // 非终端节点
    // 如果动作只有一个，直接递归其子节点，避免创建冗余的决策节点
    const auto actions = child->LegalActions();
    if (actions.size() < 2) {
      RecursiveCache(child->Child(actions[0]), prob, aidx);
      return; //这里条件返回
    }
  } else if (!save_terminals_) { // 终端状态且不保存
    histories_[idx_].outcomes_[aidx].PushBack(histories_.size(), prob); // 添加决策节点的索引和机会概率
    histories_.emplace_back(child->Returns(), idx_); // 仅保存收益
    return;
  }
  // Non-trivial decision node or terminal to be saved.
  // 找到了一个非机会节点（一般决策节点或者终端节点且需要保存），添加到当前节点的outcomes_中
  //该节点的索引是histories_.size()，本质是 idx+1
  histories_[idx_].outcomes_[aidx].PushBack(histories_.size(), prob);
  // 并将该节点添加到histories_中
  histories_.emplace_back(std::move(child), idx_);
}

void CachedDecisionPoint::CacheOutcomes() {
  // 如果当前节点是终端状态或者已经有子节点，则不需要再次缓存
  if (IsTerminal() || histories_[idx_].outcomes_.size() > 0) {
    return;
  }
  // 获取合法的动作集合。在有些游戏中，例如井字棋，下过的位置不能再下
  const auto actions = histories_[idx_].os_state_->LegalActions();
  // 遍历所有动作，缓存每个动作对应的后继节点
  for (size_t aidx = 0; aidx < actions.size(); ++aidx) {
    histories_[idx_].outcomes_.emplace_back(); // 添加一个新的Outcomes 向量
    // 以轨迹来缓存，每个状态以观测序列来标识，例如在kuhn poker中，0pb表示玩家0发的牌是J，玩家0选择了pass，玩家1选择了bet
    RecursiveCache(histories_[idx_].os_state_->Child(actions[aidx]), 1.0, aidx);
  }
}

// 从根节点开始，递归地遍历所有可能的状态，并对每个状态调用函数f
void _ForEachState(absl::flat_hash_set<std::string>& already_observed,
                   DecisionPoint& decision_point,
                   const std::function<void(const DecisionPoint&)>& f,
                   int player) {
  if (decision_point.IsTerminal()) {
    return;
  }
  const std::string info_state = decision_point.InformationStateString();
  // 如果没有被观察过，并且是指定玩家或者所有玩家
  // 将当前状态传递给函数f，并标记为已观察
  if (player < 0 || player == decision_point.PlayerToAct()) {
    if (!already_observed.contains(info_state)) {
      f(decision_point);
      already_observed.insert(std::move(info_state));
    };
  }
  // 遍历所有动作和对应的后继状态
  for (size_t a = 0; a < decision_point.NumActions(); ++a) {
    for (size_t outcome = 0; outcome < decision_point.NumOutcomes(a);
         ++outcome) {
      decision_point.Apply(a, outcome);
      _ForEachState(already_observed, decision_point, f, player);
      decision_point.Undo();
    }
  }
}

// 从根节点开始，遍历所有可能的状态，并对每个状态调用函数f
void ForEachState(DecisionPoint& root,
                  const std::function<void(const DecisionPoint&)>& f,
                  int player) {
  absl::flat_hash_set<std::string> already_observed;
  if (root.NumActions() < 2) { // 叶节点或者某些特殊节点只有一个动作，选择0，即第一个动作
    for (size_t outcome = 0; outcome < root.NumOutcomes(0); ++outcome) { // 状态是信息集的概念，一个信息集可能包含多个状态，一个动作会对应多个不同的后继状态节点
      root.Apply(0, outcome); // 应用第一个 （动作-outcome）
      _ForEachState(already_observed, root, f, player);
      root.Undo();
    }
  } else {
    _ForEachState(already_observed, root, f, player);
  }
}

// 计算从根节点出发，所有可能状态的数量
int NumStates(DecisionPoint& root, int player) {
  int count = 0;
  ForEachState(
      root, [&count](const DecisionPoint& _) { ++count; }, player);
  return count;
}

// 计算每个动作对应的状态数量
ActionMap<int> NumStatesWithAction(DecisionPoint& root, int player) {
  ActionMap<int> counts(root.NumDistinctActions(), 0);
  ForEachState(
      root,
      [&counts](const DecisionPoint& successor) {
        for (const open_spiel::Action a :
             successor.OpenSpielStatePtr()->LegalActions()) {
          ++counts[a];
        }
      },
      player);
  return counts;
}

}  // namespace hr_edl
