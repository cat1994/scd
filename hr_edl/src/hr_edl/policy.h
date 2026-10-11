#ifndef HR_EDL_POLICY_H_
#define HR_EDL_POLICY_H_

#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"
#include "hr_edl/containers.h"
#include "hr_edl/decision_point.h"
#include "hr_edl/enumeration.h"
#include "hr_edl/math.h"
#include "hr_edl/types.h"

namespace hr_edl {
// 抽象类，继承至open_spiel::Policy
// 给定状态，提供了2种获取策略的方法Response、GetStatePolicy等，分别返回vector, ActionAndProbs，
class Policy : public open_spiel::Policy {
 public:
  virtual ~Policy() = default;
  // 获取当前状态的行为策略，返回一个vector<double>
  virtual std::vector<double> Response(
      const open_spiel::State& state) const = 0;
  // 获取当前状态的行为策略，返回一个ActionsAndProbs对象 [(action_index, prob), ...], 这个是open_spiel的策略形式
  open_spiel::ActionsAndProbs GetStatePolicy(
      const open_spiel::State& state) const override final {
    const auto legal_actions = state.LegalActions();     // 获取当前状态的合法动作[...]，是{0,..., num_distinct_actions()-1}的子集
    open_spiel::ActionsAndProbs actions_and_probs;
    actions_and_probs.reserve(legal_actions.size());     // 获取当前状态的策略，返回一个vector<double>，每个动作对应一个概率
    const auto policy = Response(state);
    for (int a = 0; a < legal_actions.size(); ++a) {
      actions_and_probs.emplace_back(legal_actions[a], policy[a]);
    }
    return actions_and_probs;
  }
  virtual std::vector<double> GetPolicy(
      const std::string& info_state) const {
        std::cerr << "GetPolicy unimplemented." << std::endl;
        return std::vector<double>();
      };
  // 打印策略
  virtual void Print() const {};

};


using PolicyPtr = std::unique_ptr<Policy>;
// 获取整个游戏game所有状态的策略表，返回一个unordered_map<string, ActionsAndProbs>
// 接收参数 const open_spiel::Game& game
inline std::unordered_map<std::string, open_spiel::ActionsAndProbs>
ActionsAndProbsTable(const open_spiel::Game& game, const Policy& policy) {
  std::unordered_map<std::string, open_spiel::ActionsAndProbs> map;
  // 遍历游戏的每个状态，获取信息状态字符串，并将策略存入map
  // ForEachState遍历游戏的每个状态，调用lambda函数f
  ForEachState(*(game.NewInitialState()),
               [&map, &policy](const open_spiel::State& state) {
                 const auto iss = state.InformationStateString();
                 if (!Contains(map, iss)) {
                   map[iss] = policy.GetStatePolicy(state);
                 }
               });
  return map;
}

// 获取子树root所有状态的策略表，返回一个unordered_map<string, ActionsAndProbs>
// 接收参数 DecisionPoint& root
inline std::unordered_map<std::string, open_spiel::ActionsAndProbs>
ActionsAndProbsTable(DecisionPoint& root, const Policy& policy) {
  std::unordered_map<std::string, open_spiel::ActionsAndProbs> map;
  ForEachState(
      root,
      [&map, &policy](const DecisionPoint& dp) {
        const auto iss = dp.InformationStateString();
        if (!Contains(map, iss)) {
          map[iss] = policy.GetStatePolicy(*dp.OpenSpielStatePtr());
        }
      },
      ALL_PLAYERS);
  return map;
}


// Policy派生类，使用open_spiel::Policy指针构造
// 实现了Response方法，返回当前状态的动作概率分布向量
class ResponsePolicyAdapter : public Policy {
 public:
  ResponsePolicyAdapter(std::unique_ptr<open_spiel::Policy>&& policy)
      : policy_(std::move(policy)) {}
  // 获取当前状态的策略，返回一个vector<double>，每个动作对应一个概率
  std::vector<double> Response(
      const open_spiel::State& state) const override final {
    const auto action_prob_pairs = policy_->GetStatePolicy(state);
    std::vector<double> response(action_prob_pairs.size());
    for (int i = 0; i < action_prob_pairs.size(); ++i) {
      response[i] = action_prob_pairs[i].second;
    }
    return response;
  }

  // 获取当前状态的策略，返回一个ActionsAndProbs对象 [(action_index, prob), ...]
  open_spiel::ActionsAndProbs GetStatePolicy(
      const std::string& info_state) const override final {
    return policy_->GetStatePolicy(info_state);
  }

 private:
  std::unique_ptr<open_spiel::Policy> policy_;
};

class MapPolicy;
using MapPolicyPtr = std::unique_ptr<MapPolicy>;

// Policy派生类，使用一个unordered_map（InfoStateUvm）来存储信息集string和对应的策略vector<double>
// 它继承自Policy类，并实现了Response方法来获取当前状态的策略
// 它允许策略是行为策略或序列形式策略
class MapPolicy : public Policy {
 public:
 // 1 默认构造函数，创建一个空的策略表
  MapPolicy() : map_({}) {}
  // 2 右值策略InfoStateUvm直接构造，move语义。列表结构，方便插入
  MapPolicy(InfoStateUvm<std::vector<double>>&& map) : map_(std::move(map)) {}
  // 3 InfoStateUm需要转换为InfoStateUm，链表结构，方便查找，节省时间
  MapPolicy(const InfoStateUm<std::vector<double>>& map) {
    map_.reserve(map.size());
    for (const auto& [k, v] : map) {
      map_.emplace(k, v);
    }
  }
  // 4 左值策略InfoStateUvm直接构造
  MapPolicy(const InfoStateUvm<std::vector<double>>& map) : map_(map) {}
  // 5 通过策略Policy和root构造，这个策略是序列形式策略
  MapPolicy(const Policy& policy, DecisionPoint& root) : map_() {
    Avg(policy, root);
  }
  // 6 构造给定玩家的map_policy
  MapPolicy(const MapPolicy& map_policy, DecisionPoint& root, int player) : map_() {
    ForEachState(
        root,
        [this, player, &root, &map_policy](const DecisionPoint& dp) {
          if (dp.PlayerToAct() == player) {
            const std::string iss = dp.InformationStateString();
            auto& policy_ref = GetOrCreate<std::string, std::vector<double>>( // 获取map_中iss对应的策略引用，如果不存在则创建一个均匀分布的策略
                map_, iss,
                [&dp]() {
                  const size_t num_actions = dp.NumActions();
                  return std::vector<double>(num_actions, 1.0 / num_actions);
                });
            if (map_policy.map_.contains(iss)) {
              // 直接将map_policy中的策略拷贝到当前map_中,需要归一化
              map_[iss] = map_policy.Response(*dp.OpenSpielStatePtr());
            }
          }
        },
        player);
  }

  // 获取信息集的策略，返回一个vector<double>，每个动作对应一个概率
  // 如果信息集在map_中不存在，则返回一个均匀分布的策略
  std::vector<double> Response(
      const open_spiel::State& state) const override final {
    const std::string iss = state.InformationStateString();
    // 如果信息集在map_中存在，返回对应的策略
    if (map_.contains(iss)) {
      auto policy = map_.at(iss);
      double z = 0;
      for (const auto prob : policy) {
        z += prob;
      }
      // 如果z!=1，做安全归一化，针对序列形式策略，返回行为策略
      if (z > 1.0 || z < 1.0) {
        SafeDivide(policy, z, true);
      }
      return policy;
    } else {
      const size_t num_actions = state.LegalActions().size();
      return std::vector<double>(num_actions, 1.0 / num_actions);
    }
  }


  MapPolicyPtr Clone() const { return std::make_unique<MapPolicy>(map_); }

  // 将另一个策略的加权平均值合并到当前策略中
  void Avg(const Policy& other, DecisionPoint& root, double weight = 1.0,
           int player = ALL_PLAYERS) {
    std::vector<double> their_reach_probs(root.NumPlayers(), 1.0);
    InfoStateUvm<std::vector<double>> their_seq_probs; // map:: string, vector<double>
    for (size_t outcome_idx = 0; outcome_idx < root.NumOutcomes(0); // Artificial root node, have only one action 0
         ++outcome_idx) {
      root.Apply(0, outcome_idx); // Apply action 0 and outcome outcome_idx
      Avg_r(their_seq_probs, their_reach_probs, other, root, player); // 递归地计算他们的序列概率，并将其添加到当前的序列概率中
      root.Undo();
    }
    // 将their_seq_probs中的序列概率加权合并到当前策略map_中
    for (const auto& [iss, seq_probs] : their_seq_probs) {
      const size_t num_actions = seq_probs.size();
      auto& policy_ref = GetOrCreate<std::string, std::vector<double>>( // 获取map_中iss对应的策略引用，如果不存在则创建一个新的全0节点策略分布，即未到达的节点
          map_, iss,
          [num_actions]() { return std::vector<double>(num_actions, 0); });
      for (size_t action_idx = 0; action_idx < num_actions; ++action_idx) {
        policy_ref[action_idx] += weight * seq_probs[action_idx];
      }
    }
  }

  // 获取状态iss策略，非open_spile::state
  std::vector<double> GetPolicy(const std::string& iss) const override final {

    assert (map_.contains(iss));
      auto policy = map_.at(iss);
      double z = 0;
      for (const auto prob : policy) {
        z += prob;
      }
      // 如果z!=1，做安全归一化，针对序列形式策略，返回行为策略
      if (z > 1.0 || z < 1.0) {
        SafeDivide(policy, z, true);
      }
      return policy;

  }
  // 直接获取节点的(序列)策略,不用做归一化,适用于计算平均策略
  const std::vector<double>& GetRawPolicy(const std::string& iss) const {
    return map_.at(iss);
  }

  // 序列化打印策略
  void Print() const {
    for (const auto& [iss, policy] : map_) {
      std::cout << "InfoState: " << iss << " Policy: [";
      for (size_t a = 0; a < policy.size(); ++a) {
        std::cout << policy[a];
        if (a != policy.size() - 1) {
          std::cout << ", ";
        }
      }
      std::cout << "]" << std::endl;
    }
  }

  void Print_behavior_strategy() const {
    for (const auto& [iss, policy] : map_) {
      std::cout << "InfoState: " << iss << " Policy: [";
      // 对策略进行归一化，打印行为策略
      std::vector<double> normalized_policy = policy;
      double z = 0;
      for (const auto prob : normalized_policy) {
        z += prob;
      }
      if (z > 1.0 || z < 1.0) {
        SafeDivide(normalized_policy, z, true);
      }
      for (size_t a = 0; a < normalized_policy.size(); ++a) {
        std::cout << normalized_policy[a];
        if (a != normalized_policy.size() - 1) {
          std::cout << ", ";
        }
      }
      std::cout << "]" << std::endl;
    }
  }


 private:
 // 递归地计算他们的序列概率，并将其添加到当前的序列概率中
  // their_seq_probs: 存储他们的序列概率
  void Avg_r(InfoStateUvm<std::vector<double>>& their_seq_probs,
             std::vector<double>& their_reach_probs, const Policy& other,
             DecisionPoint& decision_point, int player) const {
    if (decision_point.IsTerminal()) {
      return;
    }
    const auto player_to_act = decision_point.PlayerToAct();
    const double their_reach_prob = their_reach_probs[player_to_act];
    const auto their_policy =
        other.Response(*decision_point.OpenSpielStatePtr());

    std::string iss = decision_point.InformationStateString();
    // 如果玩家在指定集合中（这里任意玩家都可以），且their_seq_probs中不包含该信息集, 则将其序列概率添加到their_seq_probs中
    if (PlayerInSet(player_to_act, player) && !their_seq_probs.contains(iss)) {
      std::vector<double> seq_probs;
      seq_probs.reserve(their_policy.size());
      for (size_t action_idx = 0; action_idx < their_policy.size();
           ++action_idx) {
        seq_probs.push_back(their_reach_prob * their_policy[action_idx]);
      }
      their_seq_probs.emplace(std::move(iss), std::move(seq_probs));
    }
    // 遍历所有动作和结果，递归地计算他们的序列概率
    for (size_t action_idx = 0; action_idx < their_policy.size();
         ++action_idx) {
      for (size_t outcome_idx = 0;
           outcome_idx < decision_point.NumOutcomes(action_idx);
           ++outcome_idx) {
        their_reach_probs[player_to_act] =
            their_reach_prob * their_policy[action_idx];
        decision_point.Apply(action_idx, outcome_idx);
        Avg_r(their_seq_probs, their_reach_probs, other, decision_point,
              player);
        decision_point.Undo();
      }
    }
    // 更新当前玩家下个节点的的到达概率
    their_reach_probs[player_to_act] = their_reach_prob;
  }

  // 将策略表策略清0
  void Clear() {
    for (auto& [iss, policy] : map_) {
      for (auto& prob : policy) {
        prob = 0.0;
      }
    }
  }

 private:
 // 信息集字符串iss到策略向量的映射
  mutable InfoStateUvm<std::vector<double>> map_; //mutable 可变的，private 类内部可变，外部不可变
};

// 返回一个MapPolicy对象，使用空的map {}，策略是均匀随机策略
inline MapPolicy UniformRandomPolicy() { return MapPolicy(); }

class PolicyRefProfile;
// 抽象类，玩家映射策略类，继承自Policy
// 它使用一个std::vector<size_t>来存储玩家映射关系
// 该类用于在多玩家游戏中管理不同玩家的策略
// 通过索引访问每个玩家的策略
class PlayerMapProfile : public Policy {
 public:
 // 默认玩家映射 0-N。 博弈树有默认的玩家执行顺序，这个顺序就是默认的玩家映射
  PlayerMapProfile(size_t num_players) : player_map_(Range(num_players)) {}
  // 接收右值引用 指定玩家顺序映射
  PlayerMapProfile(std::vector<size_t>&& player_map)
      : player_map_(std::move(player_map)) {}
  // 接收常量引用 指定玩家顺序映射
  PlayerMapProfile(const std::vector<size_t>& player_map)
      : player_map_(player_map) {}

  // 获取按指定玩家顺序player_map_下该状态的玩家的策略，返回一个vector<double>
  std::vector<double> Response(
      const open_spiel::State& state) const override final {
    return (*this)[state.CurrentPlayer()]->Response(state);
  }
  // 接口，重载运算符 [player], 返回*policy[player]对象
  virtual const Policy* operator[](size_t player) const = 0;

 protected:
  size_t NumPlayers() const { return player_map_.size(); }
  virtual std::vector<const Policy*> Policies() const = 0;

 protected:
  std::vector<size_t> player_map_; // 玩家映射关系
};


// 玩家映射策略类，继承自PlayerMapProfile
// 它使用一个std::vector<const Policy*>来存储玩家的策略指针
// 该类用于在多玩家游戏中管理不同玩家的策略
// 通过索引访问每个玩家的策略
// 该类不拥有策略的所有权，仅引用策略
class PolicyRefProfile : public PlayerMapProfile {
 public:
 // 1 右值引用polices, 默认玩家顺序
  PolicyRefProfile(std::vector<const Policy*>&& policies)
      : PlayerMapProfile(policies.size()), policies_(std::move(policies)) {}
  // 2 右值引用polices，指定玩家顺序
  PolicyRefProfile(std::vector<const Policy*>&& policies,
                   std::vector<size_t>&& player_map)
      : PlayerMapProfile(std::move(player_map)),
        policies_(std::move(policies)) {}
  // 3 常量引用polices, 默认玩家顺序
  PolicyRefProfile(const std::vector<const Policy*>& policies)
      : PlayerMapProfile(policies.size()), policies_(policies) {}
  // 4 常量引用polices，指定玩家顺序
  PolicyRefProfile(const std::vector<const Policy*>& policies,
                   const std::vector<size_t>& player_map)
      : PlayerMapProfile(player_map), policies_(policies) {}
  template <class PolicyLike>
  // 5 使用PolicyLike的指针构造PolicyRefProfile。默认玩家顺序
  PolicyRefProfile(const std::vector<std::unique_ptr<PolicyLike>>& policies)
      : PlayerMapProfile(policies.size()) {
    policies_.reserve(policies.size());
    for (size_t i = 0; i < policies.size(); ++i) {
      policies_.push_back(static_cast<const Policy*>(policies[i].get()));
    }
  }
  template <class PolicyLike>
  // 6 使用policyLike指针构造，默认玩家顺序，并指定玩家位置替换为learner
  PolicyRefProfile(const Policy* learner,
                   const std::vector<std::unique_ptr<PolicyLike>>& compatriots,
                   size_t learner_idx)
      : PlayerMapProfile(compatriots.size()) {
    policies_.reserve(compatriots.size());
    for (size_t player = 0; player < compatriots.size(); ++player) {
      policies_.push_back(
          (player == learner_idx)
              ? learner
              : static_cast<const Policy*>(compatriots[player].get()));
    }
  }

  // 替换player_map 中指定位置learner_idx的玩家策略为learner，并返回一个新的PolicyRefProfile对象
  PolicyRefProfile WithSubstitute(const Policy* learner,
                                  size_t learner_idx) const {
    std::vector<const Policy*> policies = Policies();
    auto player_map = player_map_;
    player_map[learner_idx] = policies.size();
    policies.push_back(learner);
    return PolicyRefProfile(std::move(policies), std::move(player_map));
  }

  // 重载运算符 [player], 返回*policy[player]对象
  const Policy* operator[](size_t player) const override {
    return policies_[player_map_[player]];
  }

 protected:
  std::vector<const Policy*> Policies() const override { return policies_; }

 private:
 // 玩家策略 Policy* 指针列表， 无所有权，仅引用
  std::vector<const Policy*> policies_;
};

// PolicyProfile是一个策略配置文件类，继承自PlayerMapProfile
// 它使用一个std::vector<PolicyPtr>来存储玩家的策略指针
// 该类用于在多玩家游戏中管理不同玩家的策略
// 通过索引访问每个玩家的策略，并提供了WithSubstitute方法来替换某个玩家的策略
// 该类拥有策略的所有权，使用智能指针 PolicyPtr
class PolicyProfile : public PlayerMapProfile {
 public:
 // 1 右值引用策略指针数组，默认玩家位置
  PolicyProfile(std::vector<PolicyPtr>&& policies)
      : PlayerMapProfile(policies.size()), policies_(std::move(policies)) {}
  // 2 右值引用策略PolicyProfile，默认玩家位置
  PolicyProfile(PolicyProfile&& profile)
      : PolicyProfile(std::move(profile.policies_)) {} // 获取profile的策略指针数组，并使用第1个构造函数
  template <class PolicyLike>
  // 3 使用PolicyLike指针构造PolicyProfile，例如learner,。默认玩家顺序
  PolicyProfile(std::vector<std::unique_ptr<PolicyLike>>&& policies)
      : PlayerMapProfile(policies.size()) {
    policies_.reserve(policies.size());
    for (size_t i = 0; i < policies.size(); ++i) {
      policies_.push_back(PolicyPtr(std::move(policies[i])));
    }
  }
  // 从MapPolicy构造PolicyProfile，默认玩家顺序
  PolicyProfile(MapPolicy& map_policy, DecisionPoint& root)
      : PlayerMapProfile(root.NumPlayers()) {
    policies_.reserve(root.NumPlayers());
    for (size_t player = 0; player < root.NumPlayers(); ++player) {
      policies_.push_back(
          PolicyPtr(new MapPolicy(map_policy, root, player)));
    }
  }

  // 替换learner_idx的玩家的策略，并返回一个新的PolicyRefProfile对象
  // 通过player_map中idx的玩家替换为num_player+1玩家,并在policies中增加策略learner
  PolicyRefProfile WithSubstitute(const Policy* learner,
                                  size_t learner_idx) const {
    std::vector<const Policy*> policies = Policies(); // 获取所有玩家的策略指针数组clone
    auto player_map = player_map_; // 复制玩家映射[0,1,2,...]
    player_map[learner_idx] = policies.size(); // 将learner放在最后一个位置 num_players+1
    policies.push_back(learner); // 添加learner到策略数组,实现添加num_players+1玩家位置的策略为更新的策略
    return PolicyRefProfile(std::move(policies), std::move(player_map)); // 返回一个新的PolicyRefProfile对象
  }
// 重载运算符
  const Policy* operator[](size_t player) const override {
    return policies_[player_map_[player]].get();
  }

  // 打印策略
  void Print() const {
    for (size_t player = 0; player < policies_.size(); ++player) {
      std::cout << "Player " << player << " Policy:" << std::endl;
      policies_[player]->Print();
    }
  }

 protected:
 // 返回所有玩家的策略指针数组clone
  std::vector<const Policy*> Policies() const override {
    std::vector<const Policy*> policies;
    policies.reserve(policies_.size());
    for (size_t i = 0; i < policies_.size(); ++i) {
      policies.push_back(policies_[i].get());// 智能指针管理的裸指针
    }
    return policies;
  }

 private:
  std::vector<PolicyPtr> policies_; // 智能指针 PolicyPtr，拥有所有权
};

}  // namespace hr_edl

#endif  // HR_EDL_POLICY_H_
