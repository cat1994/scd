#ifndef HR_EDL_DECISION_POINT_H_
#define HR_EDL_DECISION_POINT_H_

#include <unordered_map>
#include <vector>

#include "open_spiel/spiel.h"
#include "hr_edl/spiel_extra.h"

namespace hr_edl {

// A private namespace.
namespace _decision_point {

//存储某个动作后的子节点集合，包含节点索引和机会概率向量
class Outcomes {
 public:
  Outcomes() = default;
  ~Outcomes() = default;

  void PushBack(size_t idx, double prob) { // 添加决策节点的索引和机会概率
    indices_.push_back(idx);
    probs_.push_back(prob);
  }
  size_t Size() const { return indices_.size(); } // 获取决策节点的数量
  const std::vector<double>& Probs() const { return probs_; } //获取决策节点的机会概率
  size_t Idx(size_t i) const { return indices_[i]; } // 获取第i个决策节点的索引

 private:
  std::vector<size_t> indices_; // 决策节点的索引向量
  std::vector<double> probs_; // 决策节点的机会概率向量 在outcomes中存储的子节点机会节点概率从父节点到子节点的机会概率，而不是从root到子节点的累积机会概率
};

// 决策节点的缓存信息类，提供了决策节点的基本属性： state
// 状态指针，父节点索引，信息状态字符串，当前行动玩家，后继节点Outcomes，收益等
// player: 0 - num_players-1, chance: -1, terminal: -1
class HistoryCache {
 public:
  HistoryCache()  // Artificial root
      : os_state_(nullptr),
        parent_idx_(0),
        info_state_key_(),
        player_to_act_(-1),
        outcomes_(),
        returns_() {}
  HistoryCache(std::vector<double>&& returns, size_t parent_idx)  // Terminal
      : os_state_(nullptr),
        parent_idx_(parent_idx),
        info_state_key_(),
        player_to_act_(-1),
        outcomes_(),
        returns_(std::move(returns)) {
    SPIEL_CHECK_TRUE(IsTerminal()); // 确保是终端状态
  }
  HistoryCache(open_spiel::StatePtr&& os_state)  // Chance
      : os_state_(std::move(os_state)),
        parent_idx_(0),
        info_state_key_(),
        player_to_act_(-1),
        outcomes_(),
        returns_(os_state_->IsTerminal() ? os_state_->Returns()
                                         : std::vector<double>()) {
    SPIEL_CHECK_TRUE(os_state_->IsChanceNode());
  }
  HistoryCache(open_spiel::StatePtr&& os_state, size_t parent_idx) // 一般决策节点
      : os_state_(std::move(os_state)),
        parent_idx_(parent_idx),
        info_state_key_(
            os_state_->IsTerminal() ? "" : os_state_->InformationStateString()),
        player_to_act_(os_state_->IsTerminal() ? -1
                                               : os_state_->CurrentPlayer()),
        outcomes_(),
        returns_(os_state_->IsTerminal() ? os_state_->Returns()
                                         : std::vector<double>()) {
    SPIEL_CHECK_TRUE(os_state_->IsTerminal() ||
                     os_state_->CurrentPlayer() >= 0); //确保是终端节点或者玩家非机会节点
    SPIEL_CHECK_EQ(os_state_->IsTerminal(), IsTerminal()); // 确保当终端状态和判断是终端节点一致
  }
  ~HistoryCache() = default;

 public:
  bool IsTerminal() const { return !returns_.empty(); } // 是否是终端状态，通过判断返回收益是否是空

 public:
  const std::shared_ptr<const open_spiel::State> os_state_; // OpenSpiel 游戏状态指针state
  const size_t parent_idx_; // 父节点的索引
  const std::string info_state_key_; // 当前状态的key string： os_state_->InformationStateString()， 例如kuhn：0pb
  const open_spiel::Player player_to_act_; // 当前行动的玩家
  std::vector<Outcomes> outcomes_; // 决策节点Outcomes向量，每个动作对应的后继节点，Outcome包含后继节点的索引和机会概率
  const std::vector<double> returns_; // 决策节点的收益向量，只有在终端状态下才有值
};
}  // namespace _decision_point

// 抽象类，缓存节点信息的接口，提供节点的基本属性和操作，
// 动作数量，玩家数量，信息状态字符串，当前行动玩家，是否是根节点，是否是终端节点，收益等
class DecisionPoint {
 public:
  DecisionPoint(size_t num_players, size_t num_distinct_actions)
      : num_players_(num_players),
        num_distinct_actions_(num_distinct_actions) {}
  virtual ~DecisionPoint() = default;

  // Properties
  size_t NumDistinctActions() const { return num_distinct_actions_; }
  size_t NumPlayers() const { return num_players_; }

  virtual std::shared_ptr<const open_spiel::State> OpenSpielState() const = 0;
  virtual const open_spiel::State* OpenSpielStatePtr() const = 0;
  virtual const std::string& InformationStateStringRef() const = 0;
  std::string InformationStateString() const {
    return InformationStateStringRef();
  }
  virtual size_t NumActions() const = 0;
  virtual bool IsRoot() const = 0;
  virtual open_spiel::Player PlayerToAct() const = 0;
  virtual bool IsTerminal() const = 0;
  virtual const std::vector<double>& ReturnsRef() const = 0;
  std::vector<double> Returns() const { return ReturnsRef(); }
  virtual bool TerminalsAreSaved() const = 0;

  // Action-conditional properties
  virtual const std::vector<double>& OutcomeProbabilitiesRef(
      size_t action) const = 0;
  std::vector<double> OutcomeProbabilities(size_t action) const {
    return OutcomeProbabilitiesRef(action);
  }
  virtual size_t NumOutcomes(size_t action) const = 0;

  // Transformation methods
  virtual void Undo() = 0;
  virtual void UndoAll() = 0;
  virtual void Apply(size_t action, size_t outcome) = 0;
  double ApplySampledOutcome(size_t action, double random_number);

 private:
  size_t num_players_;
  size_t num_distinct_actions_;
};

// 节点实类，以向量存储了 histories_ 缓存的决策点
// 继承自DecisionPoint抽象类，实现其纯虚函数
class CachedDecisionPoint : public DecisionPoint {
 public:
 // 右值引用构造函数，接收一个OpenSpiel状态指针
  CachedDecisionPoint(open_spiel::StatePtr&& root, bool save_root = false,
                      bool save_terminals = false);
  // 左值引用构造函数，接收一个OpenSpiel状态，然后通过Clone 函数创建一个状态指针
  CachedDecisionPoint(const open_spiel::State& root, bool save_root = false,
                      bool save_terminals = false)
      : CachedDecisionPoint(root.Clone(), save_root, save_terminals) {}
// 获取信息状态字符串
  const std::string& InformationStateStringRef() const override final {
    return histories_[idx_].info_state_key_;
  }
  // 获取OpenSpiel状态指针
  std::shared_ptr<const open_spiel::State> OpenSpielState()
      const override final {
    return histories_[idx_].os_state_;
  }
  // 获取OpenSpiel状态指针
  const open_spiel::State* OpenSpielStatePtr() const override final {
    return histories_[idx_].os_state_.get();
  }
  // 获取当前决策节点的动作数量
  size_t NumActions() const override final {
    return histories_[idx_].outcomes_.size();
  }
  bool IsRoot() const override final { return idx_ == 0; }
  open_spiel::Player PlayerToAct() const override final {
    return histories_[idx_].player_to_act_;
  }
  bool IsTerminal() const override final {
    return histories_[idx_].IsTerminal();
  }
  const std::vector<double>& OutcomeProbabilitiesRef(
      size_t action) const override final {
    return histories_[idx_].outcomes_[action].Probs();
  }
  size_t NumOutcomes(size_t action) const override final {
    return histories_[idx_].outcomes_[action].Size();
  }
  const std::vector<double>& ReturnsRef() const override final {
    return histories_[idx_].returns_;
  }
  bool TerminalsAreSaved() const override final {
    return save_terminals_;
  }
  void Undo() override final { idx_ = histories_[idx_].parent_idx_; }
  void UndoAll() override final { idx_ = 0; }
  // 索引到下一个决策节点，并缓存其后继节点
  void Apply(size_t action, size_t outcome) override final {
    idx_ = histories_[idx_].outcomes_[action].Idx(outcome);
    CacheOutcomes(); // 缓存后继节点及outcomes_
  }
  size_t CurrentIdx() const { return idx_; }
 protected:
  void RecursiveCache(const open_spiel::State& child, double prob, size_t aidx);
  void RecursiveCache(open_spiel::StatePtr&& child, double prob, size_t aidx);
  void CacheOutcomes();

 private:
  size_t idx_; // 当前决策节点的索引
  std::vector<_decision_point::HistoryCache> histories_; // 决策节点的历史缓存
  bool save_terminals_;
};

// A function type for visiting decision points.
// The function takes a const reference to a DecisionPoint and returns void.
void _ForEachState(std::unordered_set<std::string>& already_observed,
                   DecisionPoint& decision_point,
                   const std::function<void(const DecisionPoint&)>& f,
                   int player);
// 从根节点开始，遍历所有可能的状态，并对每个状态调用函数f
void ForEachState(DecisionPoint& root,
                  const std::function<void(const DecisionPoint&)>& f,
                  int player);
int NumStates(DecisionPoint& root, int player = ALL_PLAYERS);
ActionMap<int> NumStatesWithAction(DecisionPoint& root, int player);

}  // namespace hr_edl

#endif  // HR_EDL_DECISION_POINT_H_
