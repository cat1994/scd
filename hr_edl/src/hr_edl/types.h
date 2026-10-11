#ifndef HR_EDL_TYPES_H_
#define HR_EDL_TYPES_H_

#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/node_hash_map.h"
#include "open_spiel/spiel.h"
#include "hr_edl/math.h"

namespace open_spiel {
using StatePtr = std::unique_ptr<open_spiel::State>;
}

namespace hr_edl {
using ActionAndProb = std::pair<open_spiel::Action, double>; // Action: int64_t
using Cfv = double;
using CfActionValues = std::vector<Cfv>;
// 记录节点每个动作的cfv值，期望值
struct CfValues {
  CfValues() = default;
  CfValues(CfValues&&) = default;
  CfValues(const CfValues&) = default;
  CfValues(std::pair<CfActionValues, Cfv>&& pair) : v_(std::move(pair.first)), ev_(pair.second) {}
  CfValues(CfActionValues&& v, Cfv ev) : v_(std::move(v)), ev_(ev) {}
  CfValues(const CfActionValues& v, Cfv ev) : v_(v), ev_(ev) {}
  // 构造函数，创建一个包含num_actions个动作的CfValues对象，动作cfv值初始化为0，期望值ev_初始化为0
  CfValues(size_t num_actions)
      : v_(std::vector<double>(num_actions, 0)), ev_(0) {}

  // 重置cfv值和期望值
  void Reset() {
    v_.assign(v_.size(), 0);
    ev_ = 0;
  }
  size_t Size() const { return v_.size(); }
  double operator()(const std::vector<double>& policy) const {
    return InnerProduct(v_, policy);
  }
  // 获取期望值
  double operator()() const { return ev_; }
  // 计算给定策略policy下的遗憾值
  double Regret(const std::vector<double>& policy) const {
    return this->operator()(policy) - ev_;
  }
  // 获取动作i的cfv值
  Cfv& operator[](size_t i) { return v_[i]; }
  // 获取cfv向量
  Cfv operator[](size_t i) const { return v_[i]; }

  CfActionValues v_; // 动作cfv值数组
  Cfv ev_; // 期望值
};

// 使用UVolMap作为absl::flat_hash_map的别名, {{key,value},...}
// 列表结构，适合查找操作多，插入删除操作少的场景
template <class... Args>
using UVolMap = absl::flat_hash_map<Args...>; 

// 使用UnorderedMap作为absl::node_hash_map的别名, {{key,value},...}
// 链表结构，适合插入删除操作多的场景
template <class... Args>
using UnorderedMap = absl::node_hash_map<Args...>; 

// 状态string map T, 列表结构, v表示Vol
template <class T>
using InfoStateUvm = UVolMap<std::string, T>; 

// 状态string map T，链表结构, 没有v表示链表结构
template <class T>
using InfoStateUm = UnorderedMap<std::string, T>; 

// 动作index map T, vector结构, 记录每个动作的值
template <class T>
using ActionMap = std::vector<T>; 

// 玩家index map T, vector结构, 记录每个玩家的值
template <class T> 
using PlayerMap = std::vector<T>; 

// 创建一个新的动作映射，返回一个ActionMap<T>对象
// num_distinct_actions()返回game的state independent动作数量,
  // 合法动作在 {0,..., num_distinct_actions()-1}
  // 例如在TicTacToe中，num_distinct_actions() = 9, 每个动作对应一个格子
template <class T>
inline ActionMap<T> NewActionMap(const open_spiel::Game& game) {
  return ActionMap<T>(game.NumDistinctActions()); 
}

// 创建一个新的玩家映射，返回一个PlayerMap<T>对象
template <class T>
inline PlayerMap<T> NewPlayerMap(const open_spiel::Game& game) {
  return PlayerMap<T>(game.NumPlayers());
}

// 创建一个玩家动作映射列表，对应每个玩家
template <class T>
inline PlayerMap<ActionMap<T>> NewPlayerActionMap(
    const open_spiel::Game& game) {
  return PlayerMap<ActionMap<T>>(game.NumPlayers(), NewActionMap<T>(game));
}

using ActionIdx = size_t;
using LegalActions = std::vector<open_spiel::Action>;
using LegalActionValues = std::vector<double>;

// [(action_index, T),...]
template <class T>
using ActionsAndValues = std::vector<std::pair<open_spiel::Action, T>>; // using Action = int64_t;

// 记录节点 CfValues 和 每个动作对应的孩子节点的key值
struct CfValueTreeNode {
  CfValueTreeNode(size_t num_actions)
      : cf_values_(num_actions), // 初始化 [0,...]
        child_keys_(num_actions, std::vector<std::string>()),// 初始化 [[keys],[keys],...]
        iwrp_(0.0) {} // 初始化重要性到达概率为0, 后续更新会累加进去，非完全信息中会多次访问该节点
  
  void Reset() { cf_values_.Reset(); iwrp_ = 0.0; }
  CfValues cf_values_; // 结构体，存储节点下每个动作的cfv值[]和期望值double
  std::vector<std::vector<std::string>> child_keys_; // 孩子节点key值
  // 重要性权重到达概率 importance_weight_reach_prob_
  double iwrp_;
};
}  // namespace hr_edl

#endif  // HR_EDL_TYPES_H_
