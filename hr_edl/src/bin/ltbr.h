#ifndef LTBR_H_
#define LTBR_H_

#include <functional>
#include <vector>

#include "absl/strings/str_format.h"
#include "hr_edl/adaptive_profile.h"
#include "hr_edl/tabular_learner.h"
#include "open_spiel/matrix_game.h"

// 定义遗憾值更新方式, 在EFR中直接给出了：next_regret:=w^t * prev_regret
// 这个函数可以扩展，例如DCFR中使用的加权平均next_regret:=alpha^t/(alpha+1)*regret^+ +
// beta^t/(beta+1)*regret^-
inline double RmUpdate(double prev_regret, double next_regret, size_t _) { return next_regret; }

// 将正的遗憾值传递给yield函数 using Enumeration Consumer = std::function<void(int, T)>;
// regrets: 遗憾值数组，yield: 函数对象，接受两个参数 yield函数还需要具体定义
inline void RmLink(const std::vector<double>& regrets,
                   const hr_edl::EnumerationConsumer<double>& yield) {
  for (size_t i = 0; i < regrets.size(); ++i) {
    yield(i, hr_edl::Relu(regrets[i]));
  }
}

struct LabeledAdaptiveProfile {
  std::string label_;
  const std::function<hr_edl::AdaptiveProfilePtr(size_t num_players, double utility_diameter)> New;
  // 创建一个新的适应性配置文件
  // New 是一个类似抽象函数， 指定了函数输入和输出格式，允许在运行时指定具体实现。
  // return: hr_edl::AdaptiveProfilePtr, para: (size_t num_players,double utility_diameter)
};

inline std::vector<std::string> LearnerProfileLabels(
    const std::vector<LabeledAdaptiveProfile>& algs) {
  std::vector<std::string> labels;
  labels.reserve(algs.size());
  for (const auto& a : algs) {
    labels.push_back(a.label_);
  }
  return labels;
}

inline std::vector<hr_edl::AdaptiveProfilePtr> NewLearnerProfiles(
    const std::vector<LabeledAdaptiveProfile>& algs, size_t num_players, double utility_diameter) {
  std::vector<hr_edl::AdaptiveProfilePtr> learners;
  learners.reserve(algs.size());
  for (const auto& a : algs) {
    learners.push_back(a.New(num_players, utility_diameter));
  }
  return learners;
}

// 定义了ltbr中的算法：id和配置
// hr_edl::AdaptiveProfilePtr
// 是hr_edl::AdaptiveProfile的智能指针，指向了其子类CfTreeLearnerProfile是AdaptiveProfile
// 该指针实例化通过：hr_edl::AdaptiveProfilePtr(new CfTreeLearnerProfile)构造函数 构造
// 构造函数输入参数   CfTreeLearnerProfile(std::vector<CfValueTreeLearnerPtr>&& learners)
// 其中BehavioralDeviationTabularCfvLearner是一个模板类，
const std::vector<LabeledAdaptiveProfile> ltbr_cfr_algs{
    // blind CF deviation
    LabeledAdaptiveProfile{
        "CFR",
        // lambda函数：不指定返回类型，c++会自己推断
        [](size_t num_players, double utility_diameter) {  // 对New 函数的具体实现，返回配置指针
          return hr_edl::AdaptiveProfilePtr(new hr_edl::CfTreeLearnerProfile(  // 实例化指针AdaptiveProfilePtr，使用了Ptr(new
                                                                               // MyClass(param1,
                                                                               // param2))
              hr_edl::BehavioralDeviationTabularCfvLearner<
                  hr_edl::ImmediateExternalSequencePredecessors>::  // 指定模板类型
                                                                    // DeviationSequencePredecessors
              NewList(num_players, RmUpdate,
                      RmLink)));  //  生成学习者learners, 用于构造类CfTreeLearnerProfile
        }},
    // 或者std::make_unique<MyClass>(param1, param2); // 直接使用make_unique创建智能指针
    // CF
    LabeledAdaptiveProfile{
        "CFR_EX+IN",
        [](size_t num_players, double utility_diameter) {
          return hr_edl::AdaptiveProfilePtr(new hr_edl::CfTreeLearnerProfile(
              hr_edl::BehavioralDeviationTabularCfvLearner<
                  hr_edl::ImmediateExInSequencePredecessors>::NewList(num_players, RmUpdate,
                                                                      RmLink)));
        }},
    // informed CF
    LabeledAdaptiveProfile{
        "CFR_IN",
        [](size_t num_players, double utility_diameter) {
          return hr_edl::AdaptiveProfilePtr(new hr_edl::CfTreeLearnerProfile(
              hr_edl::BehavioralDeviationTabularCfvLearner<
                  hr_edl::ImmediateInternalSequencePredecessors>::NewList(num_players, RmUpdate,
                                                                          RmLink)));
        }},
    // informed Action
    LabeledAdaptiveProfile{
        "A-EFR_IN",
        [](size_t num_players, double utility_diameter) {
          return hr_edl::AdaptiveProfilePtr(new hr_edl::CfTreeLearnerProfile(
              hr_edl::BehavioralDeviationTabularCfvLearner<
                  hr_edl::InformedActionSequencePredecessors>::NewList(num_players, RmUpdate,
                                                                       RmLink)));
        }},
    // BPS
    LabeledAdaptiveProfile{
        "BPS-EFR",
        [](size_t num_players, double utility_diameter) {
          return hr_edl::AdaptiveProfilePtr(new hr_edl::CfTreeLearnerProfile(
              hr_edl::BehavioralDeviationTabularCfvLearner<
                  hr_edl::BlindPartialSequencePredecessors>::NewList(num_players, RmUpdate,
                                                                     RmLink)));
        }},
    // TIPS
    LabeledAdaptiveProfile{
        "TIPS-EFR",
        [](size_t num_players, double utility_diameter) {
          return hr_edl::AdaptiveProfilePtr(new hr_edl::CfTreeLearnerProfile(
              hr_edl::BehavioralDeviationTabularCfvLearner<
                  hr_edl::TwiceInformedPartialSequencePredecessors>::NewList(num_players, RmUpdate,
                                                                             RmLink)));
        }},
    // CSPS
    LabeledAdaptiveProfile{
        "CSPS-EFR",
        [](size_t num_players, double utility_diameter) {
          return hr_edl::AdaptiveProfilePtr(new hr_edl::CfTreeLearnerProfile(
              hr_edl::BehavioralDeviationTabularCfvLearner<
                  hr_edl::CausalPartialSequencePredecessors>::NewList(num_players, RmUpdate,
                                                                      RmLink)));
        }},
    // CFPS
    LabeledAdaptiveProfile{
        "CFPS-EFR",
        [](size_t num_players, double utility_diameter) {
          return hr_edl::AdaptiveProfilePtr(new hr_edl::CfTreeLearnerProfile(
              hr_edl::BehavioralDeviationTabularCfvLearner<
                  hr_edl::CounterfactualPartialSequencePredecessors>::NewList(num_players, RmUpdate,
                                                                              RmLink)));
        }},
    // CFPS_Ex+In
    LabeledAdaptiveProfile{
        "CFPS-EFR_EX+IN",
        [](size_t num_players, double utility_diameter) {
          return hr_edl::AdaptiveProfilePtr(new hr_edl::CfTreeLearnerProfile(
              hr_edl::BehavioralDeviationTabularCfvLearner<
                  hr_edl::CounterfactualPartialSequenceExInPredecessors>::NewList(num_players,
                                                                                  RmUpdate,
                                                                                  RmLink)));
        }},
    // TICS_EX+IN
    LabeledAdaptiveProfile{
        "TIPS-EFR_EX+IN", [](size_t num_players, double utility_diameter) {
          return hr_edl::AdaptiveProfilePtr(new hr_edl::CfTreeLearnerProfile(
              hr_edl::BehavioralDeviationTabularCfvLearner<
                  hr_edl::TwiceInformedPartialSequenceExInPredecessors>::NewList(num_players,
                                                                                 RmUpdate,
                                                                                 RmLink)));
        }}};

const std::vector<LabeledAdaptiveProfile> expensive_cfr_algs{
    // BHV
    LabeledAdaptiveProfile{
        "BEHAV-EFR", [](size_t num_players, double utility_diameter) {
          return hr_edl::AdaptiveProfilePtr(new hr_edl::CfTreeLearnerProfile(
              hr_edl::BehavioralDeviationTabularCfvLearner<hr_edl::BehavioralPredecessors>::NewList(
                  num_players, RmUpdate, RmLink)));
        }}};

// inline std::vector<LabeledAdaptiveProfile> rt_cfr_algs(double rt_cfr_weight,
//                                                         size_t rt_cfr_T) {
//   return {LabeledAdaptiveProfile{
//       "RT_CFR",
//       [rt_cfr_weight, rt_cfr_T](size_t num_players, double utility_diameter) {
//         return hr_edl::AdaptiveProfilePtr(new hr_edl::RTCfTreeLearnerProfile(
//             hr_edl::BehavioralDeviationTabularCfvLearner<
//                 hr_edl::ImmediateExternalSequencePredecessors>::NewList(
//                 num_players, RmUpdate, RmLink), rt_cfr_weight, rt_cfr_T));
//       }}};
// }

inline std::vector<LabeledAdaptiveProfile> ltbr_perturbed_cfr_algs(double epsilon) {
  return {
      LabeledAdaptiveProfile{
          "A-EFR_IN",
          [epsilon](size_t num_players, double utility_diameter) {
            return hr_edl::AdaptiveProfilePtr(new hr_edl::PerturbedCfTreeLearnerProfile(
                hr_edl::BehavioralDeviationTabularCfvLearner<
                    hr_edl::InformedActionSequencePredecessors>::NewList(num_players, RmUpdate,
                                                                         RmLink),
                epsilon));
          }},
      LabeledAdaptiveProfile{
          "CSPS-EFR",
          [epsilon](size_t num_players, double utility_diameter) {
            return hr_edl::AdaptiveProfilePtr(new hr_edl::PerturbedCfTreeLearnerProfile(
                hr_edl::BehavioralDeviationTabularCfvLearner<
                    hr_edl::CausalPartialSequencePredecessors>::NewList(num_players, RmUpdate,
                                                                        RmLink),
                epsilon));
          }},
      LabeledAdaptiveProfile{
          "CFPS-EFR",
          [epsilon](size_t num_players, double utility_diameter) {
            return hr_edl::AdaptiveProfilePtr(new hr_edl::PerturbedCfTreeLearnerProfile(
                hr_edl::BehavioralDeviationTabularCfvLearner<
                    hr_edl::CounterfactualPartialSequencePredecessors>::NewList(num_players,
                                                                                RmUpdate, RmLink),
                epsilon));
          }},
      LabeledAdaptiveProfile{
          "CFR",
          [epsilon](size_t num_players, double utility_diameter) {
            return hr_edl::AdaptiveProfilePtr(new hr_edl::PerturbedCfTreeLearnerProfile(
                hr_edl::BehavioralDeviationTabularCfvLearner<
                    hr_edl::CounterfactualPartialSequenceExInPredecessors>::NewList(num_players,
                                                                                    RmUpdate,
                                                                                    RmLink),
                epsilon));
          }},
      LabeledAdaptiveProfile{
          "CFR_IN",
          [epsilon](size_t num_players, double utility_diameter) {
            return hr_edl::AdaptiveProfilePtr(new hr_edl::PerturbedCfTreeLearnerProfile(
                hr_edl::BehavioralDeviationTabularCfvLearner<
                    hr_edl::TwiceInformedPartialSequencePredecessors>::NewList(num_players,
                                                                               RmUpdate, RmLink),
                epsilon));
          }},

  };
}

// 配套pbce测试的其他算法，包括CSPS，CF，CF_In
const std::vector<LabeledAdaptiveProfile> ltbr_cfr_pbce_full_algs{
    // blind CF deviation
    LabeledAdaptiveProfile{
        "CFR",
        // lambda函数：不指定返回类型，c++会自己推断
        [](size_t num_players, double utility_diameter) {  // 对New 函数的具体实现，返回配置指针
          return hr_edl::AdaptiveProfilePtr(new hr_edl::CfTreeLearnerProfile(  // 实例化指针AdaptiveProfilePtr，使用了Ptr(new
                                                                               // MyClass(param1,
                                                                               // param2))
              hr_edl::BehavioralDeviationTabularCfvLearner<
                  hr_edl::ImmediateExternalSequencePredecessors>::  // 指定模板类型
                                                                    // DeviationSequencePredecessors
              NewList(num_players, RmUpdate,
                      RmLink)));  //  生成学习者learners, 用于构造类CfTreeLearnerProfile
        }},
    // 或者std::make_unique<MyClass>(param1, param2); // 直接使用make_unique创建智能指针
    // informed CF
    LabeledAdaptiveProfile{
        "CFR_IN",
        [](size_t num_players, double utility_diameter) {
          return hr_edl::AdaptiveProfilePtr(new hr_edl::CfTreeLearnerProfile(
              hr_edl::BehavioralDeviationTabularCfvLearner<
                  hr_edl::ImmediateInternalSequencePredecessors>::NewList(num_players, RmUpdate,
                                                                          RmLink)));
        }},
    // CSPS
    LabeledAdaptiveProfile{
        "CSPS-EFR",
        [](size_t num_players, double utility_diameter) {
          return hr_edl::AdaptiveProfilePtr(new hr_edl::CfTreeLearnerProfile(
              hr_edl::BehavioralDeviationTabularCfvLearner<
                  hr_edl::CausalPartialSequencePredecessors>::NewList(num_players, RmUpdate,
                                                                      RmLink)));
        }},
};

inline std::vector<LabeledAdaptiveProfile> qbce_cfr_algs(size_t query_budget) {
  return {
      LabeledAdaptiveProfile{
          "QB-EFR",
          [query_budget](size_t num_players, double utility_diameter) {
            return hr_edl::AdaptiveProfilePtr(new hr_edl::QBCECfTreeLearnerProfile(
                hr_edl::BehavioralDeviationTabularCfvLearner<hr_edl::QueryPredecessors>::NewList(
                    num_players, RmUpdate, RmLink, query_budget, false, false)));
          }},
      LabeledAdaptiveProfile{
          "QB-EFR_SEQ",
          [query_budget](size_t num_players, double utility_diameter) {
            return hr_edl::AdaptiveProfilePtr(new hr_edl::QBCECfTreeLearnerProfile(
                hr_edl::BehavioralDeviationTabularCfvLearner<hr_edl::QueryPredecessors>::NewList(
                    num_players, RmUpdate, RmLink, query_budget, true, false)));
          }},
      LabeledAdaptiveProfile{
          "QB-EFR_CS",
          [query_budget](size_t num_players, double utility_diameter) {
            return hr_edl::AdaptiveProfilePtr(new hr_edl::QBCECfTreeLearnerProfile(
                hr_edl::BehavioralDeviationTabularCfvLearner<hr_edl::QueryPredecessors>::NewList(
                    num_players, RmUpdate, RmLink, query_budget, false, true)));
          }},
      LabeledAdaptiveProfile{
          "QB-EFR_SEQ+CS", [query_budget](size_t num_players, double utility_diameter) {
            return hr_edl::AdaptiveProfilePtr(new hr_edl::QBCECfTreeLearnerProfile(
                hr_edl::BehavioralDeviationTabularCfvLearner<hr_edl::QueryPredecessors>::NewList(
                    num_players, RmUpdate, RmLink, query_budget, true, true)));
          }}};
}

// qbc: 包含了<=k的结构
inline std::vector<LabeledAdaptiveProfile> qbce_cfr_algs_(size_t query_budget) {
  return {
      LabeledAdaptiveProfile{
          "QBC-EFR",
          [query_budget](size_t num_players, double utility_diameter) {
            return hr_edl::AdaptiveProfilePtr(new hr_edl::QBCECfTreeLearnerProfile(
                hr_edl::BehavioralDeviationTabularCfvLearner<hr_edl::QueryPredecessors_>::NewList(
                    num_players, RmUpdate, RmLink, query_budget, false, false)));
          }},
      LabeledAdaptiveProfile{
          "QBC-EFR_SEQ",
          [query_budget](size_t num_players, double utility_diameter) {
            return hr_edl::AdaptiveProfilePtr(new hr_edl::QBCECfTreeLearnerProfile(
                hr_edl::BehavioralDeviationTabularCfvLearner<hr_edl::QueryPredecessors_>::NewList(
                    num_players, RmUpdate, RmLink, query_budget, true, false)));
          }},
      LabeledAdaptiveProfile{
          "QBC-EFR_CS",
          [query_budget](size_t num_players, double utility_diameter) {
            return hr_edl::AdaptiveProfilePtr(new hr_edl::QBCECfTreeLearnerProfile(
                hr_edl::BehavioralDeviationTabularCfvLearner<hr_edl::QueryPredecessors_>::NewList(
                    num_players, RmUpdate, RmLink, query_budget, false, true)));
          }},
      LabeledAdaptiveProfile{
          "QBC-EFR_SEQ+CS", [query_budget](size_t num_players, double utility_diameter) {
            return hr_edl::AdaptiveProfilePtr(new hr_edl::QBCECfTreeLearnerProfile(
                hr_edl::BehavioralDeviationTabularCfvLearner<hr_edl::QueryPredecessors_>::NewList(
                    num_players, RmUpdate, RmLink, query_budget, true, true)));
          }}};
}

// 临时测试组 CF， BHV
inline std::vector<LabeledAdaptiveProfile> ltbr_cfr_bhv_algs{
    // blind CF deviation
    LabeledAdaptiveProfile{
        "CFR",
        // lambda函数：不指定返回类型，c++会自己推断
        [](size_t num_players, double utility_diameter) {  // 对New 函数的具体实现，返回配置指针
          return hr_edl::AdaptiveProfilePtr(new hr_edl::CfTreeLearnerProfile(  // 实例化指针AdaptiveProfilePtr，使用了Ptr(new
                                                                               // MyClass(param1,
                                                                               // param2))
              hr_edl::BehavioralDeviationTabularCfvLearner<
                  hr_edl::ImmediateExternalSequencePredecessors>::  // 指定模板类型
                                                                    // DeviationSequencePredecessors
              NewList(num_players, RmUpdate,
                      RmLink)));  //  生成学习者learners, 用于构造类CfTreeLearnerProfile
        }},
    // 或者std::make_unique<MyClass>(param1, param2); // 直接使用make_unique创建智能指针
    // BHV
    LabeledAdaptiveProfile{
        "BEHAV-EFR", [](size_t num_players, double utility_diameter) {
          return hr_edl::AdaptiveProfilePtr(new hr_edl::CfTreeLearnerProfile(
              hr_edl::BehavioralDeviationTabularCfvLearner<hr_edl::BehavioralPredecessors>::NewList(
                  num_players, RmUpdate, RmLink)));
        }}};

inline std::vector<LabeledAdaptiveProfile> AlgsInGroup(size_t alg_group, size_t query_budget = 0,
                                                       double rt_cfr_weight = 0.0,
                                                       size_t rt_cfr_T = 1, double epsilon = 0.0) {
  std::vector<LabeledAdaptiveProfile> algs;
  // 根据k生成不同k-的QBCE算法,最多4个,包含1, 2, 4, 8-QBCE算法
  std::vector<size_t> qbce_k_groups;
  for (size_t k = 1; k <= 8; k *= 2) {
    qbce_k_groups.push_back(k);
    if (k >= query_budget) {
      break;
    }
  }
  // {1, 2, 4, 8}; //
  // 定义qbce_mode的alg_group值，例如6表示测试不同k-的QBCE，7表示测试不同k-的SEQ-QBCE
  if (alg_group == 0 or alg_group == 1) {  // add ltbr algs
    for (const auto& alg : ltbr_cfr_algs) {
      algs.push_back(alg);
    }
  }
  if (alg_group == 1) {  // add expensive algs
    for (const auto& alg : expensive_cfr_algs) {
      algs.push_back(alg);
    }
  }
  // if (alg_group == 2) { // add rt-cfr algs
  //   for (const auto& alg : rt_cfr_algs(rt_cfr_weight, rt_cfr_T)) {
  //     algs.push_back(alg);
  //   }
  // }
  if (alg_group == 3) {  // add perturbed ltbr algs
    for (const auto& alg : ltbr_perturbed_cfr_algs(epsilon)) {
      algs.push_back(alg);
    }
  }

  // NOTE：QBCE
  if (alg_group == 4) {  // add qbce algs
    for (auto& alg : qbce_cfr_algs(query_budget)) {
      // 替换label为不同k-的形式，例如1-QB-EFR, 2-QB-EFR, ...
      if (absl::StrContains(alg.label_, "QB-EFR")) {
        alg.label_ = absl::StrFormat("%d-%s", query_budget, alg.label_);
      }
      algs.push_back(alg);
    }
  }

  // CSPS, CF, CF_IN, BHV, k-QB-EFR, K-QB-EFR_SEQ, K-QB-EFR_CS, K-QB-EFR_SEQ+CS
  if (alg_group == 5) {
    for (const auto& alg : ltbr_cfr_pbce_full_algs) {
      algs.push_back(alg);
    }
    for (const auto& alg : expensive_cfr_algs) {
      algs.push_back(alg);
    }
    for (auto& alg : qbce_cfr_algs(query_budget)) {
      // 替换label为不同k-的形式，例如1-QB-EFR, 2-QB-EFR, ...
      if (absl::StrContains(alg.label_, "QB-EFR")) {
        alg.label_ = absl::StrFormat("%d-%s", query_budget, alg.label_);
      }
      algs.push_back(alg);
    }
  }

  // 只选择QB-EFR算法，并且替换label为不同k-的形式，例如1-QB-EFR, 2-QB-EFR, ...
  if (alg_group == 6) {  //
    for (const size_t k : qbce_k_groups) {
      for (auto& alg : qbce_cfr_algs(k)) {
        if (alg.label_ == "QB-EFR") {
          alg.label_ = absl::StrFormat("%d-%s", k, alg.label_);
          algs.push_back(alg);
        }
      }
    }
  }
  // 只选择QB-EFR-SEQ算法，并且替换label为不同k-的形式，例如1-QB-EFR_SEQ, 2-QB-EFR_SEQ, ...
  if (alg_group == 7) {
    for (const size_t k : qbce_k_groups) {
      for (auto& alg : qbce_cfr_algs(k)) {
        if (alg.label_ == "QB-EFR_SEQ") {
          alg.label_ = absl::StrFormat("%d-%s", k, alg.label_);
          algs.push_back(alg);
        }
      }
    }
  }

  // 只选择QB-EFR-CS算法，并且替换label为不同k-的形式，例如1-QB-EFR_CS, 2-QB-EFR_CS, ...
  if (alg_group == 8) {
    for (const size_t k : qbce_k_groups) {
      for (auto& alg : qbce_cfr_algs(k)) {
        if (alg.label_ == "QB-EFR_CS") {
          alg.label_ = absl::StrFormat("%d-%s", k, alg.label_);
          algs.push_back(alg);
        }
      }
    }
  }
  // 只选择QB-EFR-SEQ+CS算法，并且替换label为不同k-的形式，例如1-QB-EFR_SEQ+CS, 2-QB-EFR_SEQ+CS, ...
  if (alg_group == 9) {
    for (const size_t k : qbce_k_groups) {
      for (auto& alg : qbce_cfr_algs(k)) {
        if (alg.label_ == "QB-EFR_SEQ+CS") {
          alg.label_ = absl::StrFormat("%d-%s", k, alg.label_);
          algs.push_back(alg);
        }
      }
    }
  }

  // 总体测试:  1-QB-EFR、
  // {2, 4} - {QB-EFR, -QB-EFR_SEQ, -QB-EFR_CS, -QB-EFR_SEQ+CS}
  // ACT, CF/0-QB-EFR, CF_IN, CFPS-EFR、BHV/infty-QB-EFR算法

  // Select the seven SCD main methods before learners are instantiated.
  if (alg_group == 11) {
    const auto full_group = AlgsInGroup(10, query_budget, rt_cfr_weight,
                                      rt_cfr_T, epsilon);
    for (const char* label : {"CFR", "1-QB-EFR", "2-QB-EFR", "4-QB-EFR",
                              "A-EFR_IN", "CSPS-EFR", "BEHAV-EFR"}) {
      for (const auto& alg : full_group) {
        if (alg.label_ == label) algs.push_back(alg);
      }
    }
  }

  if (alg_group == 10) {
    for (const auto& alg : ltbr_cfr_algs) {
      // ACT
      if (alg.label_ == "A-EFR_IN") {
        algs.push_back(alg);
      }
      // CFR/0-QB-EFR
      if (alg.label_ == "CFR") {
        algs.push_back(alg);
      }
      // CF_IN
      if (alg.label_ == "CFR_IN") {
        algs.push_back(alg);
      }
      // CSPS-EFR
      if (alg.label_ == "CSPS-EFR") {
        algs.push_back(alg);
      }
    }
    // BHV
    for (const auto& alg : expensive_cfr_algs) {
      algs.push_back(alg);
    }
    // assert (!qbce_k_groups.empty());
    //  1-QB-EFR
    if (query_budget >= 1) {
      for (auto& alg : qbce_cfr_algs(1)) {
        if (alg.label_ == "QB-EFR") {
          alg.label_ = absl::StrFormat("%d-%s", 1, alg.label_);
          algs.push_back(alg);
        }
      }
    }

    // 2/4 -QB-EFR, -QB-EFR_SEQ, -QB-EFR_CS, -QB-EFR_SEQ+CS
    for (auto k : {2, 4}) {
      if (k > query_budget) continue;
      for (auto& alg : qbce_cfr_algs(k)) {
        if (alg.label_ == "QB-EFR" || alg.label_ == "QB-EFR_SEQ" || alg.label_ == "QB-EFR_CS" ||
            alg.label_ == "QB-EFR_SEQ+CS") {
          alg.label_ = absl::StrFormat("%d-%s", k, alg.label_);
          algs.push_back(alg);
        }
      }
    }
  }

  // test qbc和qb的区别
  if (alg_group == 99) {
    // for (const auto& alg : expensive_cfr_algs) {
    //   algs.push_back(alg);
    // }
    for (auto& alg : qbce_cfr_algs(query_budget)) {
      if (alg.label_ == "QB-EFR") {
        alg.label_ = absl::StrFormat("%d-%s", query_budget, alg.label_);
        algs.push_back(alg);
      }
    }
    for (auto& alg : qbce_cfr_algs_(query_budget)) {
      if (alg.label_ == "QBC-EFR") {
        alg.label_ = absl::StrFormat("%d-%s", query_budget, alg.label_);
        algs.push_back(alg);
      }
    }
  }

  if (alg_group == 100) {
    for (const auto& alg : ltbr_cfr_bhv_algs) {
      algs.push_back(alg);
    }
  }
  return algs;
}

#endif  // LTBR_H_
