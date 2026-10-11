#include "bin/ltbr.h"
#include "hr_edl/stopwatch.h"

#include <iostream>
#include <map>
#include <numeric>
#include <stdexcept>

namespace {
using History = std::vector<int>;  // -1 means an unspecified coordinate.
struct Expert {
  History history;
  int source;  // -1 for EX, otherwise IN(source -> target).
  int target;
  bool operator<(const Expert& x) const {
    return std::tie(history, source, target) < std::tie(x.history, x.source, x.target);
  }
};
void Close(double x, double y, const char* label) {
  if (std::abs(x - y) > 1e-9 * (1 + std::abs(x) + std::abs(y))) {
    std::cerr << label << ": " << x << " != " << y << '\n';
    throw std::runtime_error(label);
  }
}
size_t Length(const History& h) {
  return std::count_if(h.begin(), h.end(), [](int a) { return a >= 0; });
}
double ProductWeight(const History& h, const std::vector<std::vector<double>>& p) {
  double weight = 1;
  for (size_t j = 0; j < h.size(); ++j) if (h[j] >= 0) weight *= p[j][h[j]];
  return weight;
}
std::vector<History> Enumerate(const std::vector<size_t>& actions, size_t k) {
  std::vector<History> histories(1);
  for (size_t j = 0; j < actions.size(); ++j) {
    const auto previous = histories;
    for (auto& h : histories) h.push_back(-1);
    for (const auto& h : previous) {
      if (Length(h) == k) continue;
      for (size_t a = 0; a < actions[j]; ++a) {
        auto child = h; child.push_back(a); histories.push_back(child);
      }
    }
  }
  return histories;
}
// Fixed extension coordinates depend only on the expert, never on the round.
std::vector<History> Extend(History h, size_t length, const std::vector<size_t>& actions) {
  if (Length(h) == length) return {h};
  const auto j = std::find(h.begin(), h.end(), -1) - h.begin();
  std::vector<History> result;
  for (size_t a = 0; a < actions[j]; ++a) {
    auto child = h; child[j] = a;
    const auto tails = Extend(child, length, actions);
    result.insert(result.end(), tails.begin(), tails.end());
  }
  return result;
}

void Coverage(size_t d, size_t k) {
  const size_t n = 3;  // Test more than two current actions.
  std::vector<size_t> actions(d);
  for (size_t j = 0; j < d; ++j) actions[j] = 2 + j % 2;
  const auto histories = Enumerate(actions, k);
  std::vector<Expert> retained, full;
  std::map<Expert, std::vector<Expert>> coefficients;
  for (const auto& h : histories) {
    for (int b = 0; b < n; ++b) {
      const Expert ex{h, -1, b};
      full.push_back(ex);
      if (Length(h) == k) retained.push_back(ex);
      if (k == 0 || d >= k) {
        for (const auto& extension : Extend(h, k, actions))
          coefficients[ex].push_back({extension, -1, b});
      } else {
        for (const auto& extension : Extend(h, d, actions))
          for (int a = 0; a < n; ++a) if (a != b)
            coefficients[ex].push_back({extension, a, b});
      }
      if (k > 0 && Length(h) < k) {
        for (int a = 0; a < n; ++a) if (a != b) {
          const Expert in{h, a, b}; full.push_back(in);
          const size_t layer = std::min(k - 1, d);
          if (Length(h) == layer) retained.push_back(in);
          for (const auto& extension : Extend(h, layer, actions))
            coefficients[in].push_back({extension, a, b});
        }
      }
    }
  }
  size_t kappa = 0;
  for (const auto& entry : coefficients) {
    kappa = std::max(kappa, entry.second.size());
    for (const auto& term : entry.second)
      if (std::find_if(retained.begin(), retained.end(), [&term](const Expert& x) {
        return !(x < term) && !(term < x);
      }) == retained.end()) throw std::runtime_error("coefficient outside retained set");
  }
  std::map<Expert, double> sums;
  const auto external = hr_edl::SwapActionTranformation::External(n);
  const auto internal = hr_edl::SwapActionTranformation::Internal(n);
  for (int round = 0; round < 3; ++round) {
    std::vector<std::vector<double>> pred(d);
    for (size_t j = 0; j < d; ++j) {
      pred[j] = actions[j] == 2 ? (round == 0 ? std::vector<double>{0, 1} :
        round == 1 ? std::vector<double>{.3, .7} : std::vector<double>{.8, .2}) :
        (round == 0 ? std::vector<double>{.2, 0, .8} :
        round == 1 ? std::vector<double>{.5, .4, .1} : std::vector<double>{0, .6, .4});
    }
    // Bind product weights and retained layers to the actual C++ propagation.
    hr_edl::QueryPredecessors compressed(k, false, false);
    hr_edl::QueryPredecessors_ complete(k, false, false);
    std::vector<double> weights{1}; std::vector<size_t> budgets{k};
    for (size_t j = 0; j < d; ++j) {
      auto next = compressed.SuccessorReachProbs(weights, pred[j].data(), actions[j], 0, budgets);
      auto full_next = complete.SuccessorReachProbs(weights, pred[j].data(), actions[j], 0, budgets);
      if (next != full_next) throw std::runtime_error("successor implementation mismatch");
      weights = next.first; budgets = next.second;
    }
    if (weights.size() != histories.size()) throw std::runtime_error("history count mismatch");
    std::vector<double> ex_weights, in_weights, full_in_weights;
    for (size_t h = 0; h < histories.size(); ++h) {
      const auto length = Length(histories[h]);
      Close(weights[h], ProductWeight(histories[h], pred), "coordinate product");
      if (budgets[h] != k - length) throw std::runtime_error("remaining budget mismatch");
      if (length == k) ex_weights.push_back(weights[h]);
      if (k > 0 && length == std::min(k - 1, d)) in_weights.push_back(weights[h]);
      if (length < k) full_in_weights.push_back(weights[h]);
      if (length < k) {
        for (size_t j = 0; j < d; ++j) if (histories[h][j] < 0) {
          double refined = 0;
          for (size_t a = 0; a < actions[j]; ++a) {
            auto extension = histories[h]; extension[j] = a;
            const auto index = std::find(histories.begin(), histories.end(), extension) - histories.begin();
            if (index == histories.size()) throw std::runtime_error("missing refinement history");
            refined += weights[index];
          }
          Close(weights[h], refined, "one-coordinate weight refinement");
        }
      }
    }
    if (compressed.ExternalPredecessorReachProbs(weights, budgets) != ex_weights ||
        compressed.InternalPredecessorReachProbs(weights, budgets) != in_weights ||
        complete.ExternalPredecessorReachProbs(weights, budgets) != weights ||
        complete.InternalPredecessorReachProbs(weights, budgets) != full_in_weights)
      throw std::runtime_error("actual EX/IN layer mismatch");

    const std::vector<double> p = round == 0 ? std::vector<double>{0, .4, .6} :
      round == 1 ? std::vector<double>{.2, .5, .3} : std::vector<double>{.8, .2, 0};
    const std::vector<double> v = round == 0 ? std::vector<double>{3, -2, 1} :
      round == 1 ? std::vector<double>{-1, 4, 0} : std::vector<double>{2, -3, 5};
    const double baseline = std::inner_product(p.begin(), p.end(), v.begin(), 0.0);
    hr_edl::CfValues cfv(v, baseline);
    for (int b = 0; b < n; ++b) {
      double sum = 0;
      for (const auto& phi : internal) {
        for (int a = 0; a < n; ++a) if (a != b && phi.SwapTarget(a) == b) {
          Close(phi.Regret(cfv, p), p[a] * (v[b] - v[a]), "IN formula");
          sum += phi.Regret(cfv, p);
        }
      }
      Close(external[b].Regret(cfv, p), v[b] - baseline, "EX formula");
      Close(external[b].Regret(cfv, p), sum, "EX=sum IN");
    }
    for (const auto& e : full) {
      const double instantaneous = e.source < 0 ? v[e.target] - baseline :
        p[e.source] * (v[e.target] - v[e.source]);
      sums[e] += ProductWeight(e.history, pred) * instantaneous;
    }
    double retained_max = 0, full_max = 0, bound = 0;
    for (const auto& e : retained) retained_max = std::max(retained_max, sums[e]);
    for (const auto& e : full) {
      double reconstructed = 0, positive_sum = 0;
      for (const auto& term : coefficients[e]) {
        reconstructed += sums[term]; positive_sum += std::max(0.0, sums[term]);
      }
      Close(sums[e], reconstructed, "same-trajectory cumulative reconstruction");
      full_max = std::max(full_max, sums[e]); bound = std::max(bound, positive_sum);
    }
    if (full_max > bound + 1e-9 || bound > kappa * retained_max + 1e-9)
      throw std::runtime_error("coverage bound violated");
  }
  std::cout << "local coverage d=" << d << " k=" << k << " kappa=" << kappa << '\n';
}

struct TestDecision : hr_edl::_tl::ImmediateDecisionInfo {
  TestDecision() : ImmediateDecisionInfo(2, 1, 0) {}
  void SetPolicy() { policy_ = {1, 0}; }
};
void RawCfrPlus() {
  TestDecision decision;
  decision.SetCfrPlus(true);
  const hr_edl::_tl::RegretTransformation update = [](double, double r, size_t) { return r; };
  const hr_edl::_tl::LinkFn link = [](const auto& r, const auto& emit) {
    for (size_t i = 0; i < r.size(); ++i) emit(i, std::max(0.0, r[i]));
  };
  const hr_edl::_tl::ReachProbLists weights{{1}, {1}, {}}, empty{{}, {}, {}};
  for (double value : {-2.0, 1.0}) {
    decision.SetPolicy();
    decision.Update(hr_edl::CfValues({0, value}, 0), update, link, weights, empty);
  }
  Close(std::get<0>(decision.PhiRegrets()), 1, "CFR+ solver truncation");
  Close(decision.StageOriginalLocalRegret(), 0, "untruncated original accumulation");
  decision.ResetStageEvaluation();
  Close(decision.StageOriginalLocalRegret(), 0, "stage reset");
  Close(std::get<0>(decision.PhiRegrets()), 1, "stage reset preserves solver");
}
struct FakeClock {
  using duration = std::chrono::nanoseconds;
  using time_point = std::chrono::time_point<FakeClock>;
  static inline duration elapsed{};
  static time_point now() { return time_point(elapsed); }
};

void SamePolicies(hr_edl::DecisionPoint& root, const hr_edl::Policy& x,
                  const hr_edl::Policy& y) {
  hr_edl::ForEachState(root, [&](const hr_edl::DecisionPoint& point) {
    if (point.IsTerminal()) return;
    const auto a = x.Response(*point.OpenSpielStatePtr());
    const auto b = y.Response(*point.OpenSpielStatePtr());
    if (a.size() != b.size()) throw std::runtime_error("policy action count");
    for (size_t i = 0; i < a.size(); ++i) Close(a[i], b[i], "fixed column policy");
  }, hr_edl::ALL_PLAYERS);
}
void FixedSemantics() {
  const auto game = open_spiel::LoadGame("kuhn_poker(players=2)");
  hr_edl::CachedDecisionPoint root(game->NewInitialState(), false, true);
  auto sampler = hr_edl::NewSampler("null", 0);
  for (size_t k : {1, 2, 4}) {
    const auto algorithms = AlgsInGroup(99, k);
    auto columns = NewLearnerProfiles(algorithms, 2, 4);
    auto reference = NewLearnerProfiles(algorithms, 2, 4);
    std::vector<std::vector<hr_edl::AdaptiveProfilePtr>> rows;
    for (size_t col = 0; col < 2; ++col) rows.push_back(NewLearnerProfiles(algorithms, 2, 4));
    for (size_t col = 0; col < 2; ++col) {
      for (int t = 0; t < 2; ++t) {
        const auto frozen = columns[col]->Frozen();
        SamePolicies(root, frozen, reference[col]->Frozen());
        for (auto& row : rows[col]) {
          for (int player = 0; player < 2; ++player) {
            hr_edl::PolicyCfValueTreeEvaluator evaluator(player);
            const auto evaluation = evaluator.ComputeCfValueTreeEvaluation(
                root, frozen.WithSubstitute(row->Strategy(player), player), *sampler);
            const auto& tree = *evaluation.cfv_nodes_;
            hr_edl::ForEachState(root, [&](const hr_edl::DecisionPoint& point) {
              if (point.IsTerminal() || point.PlayerToAct() != player) return;
              const auto& values = tree.at(point.InformationStateString()).cf_values_;
              const auto policy = row->Strategy(player)->Response(*point.OpenSpielStatePtr());
              Close(values(), values(policy), "actual cfv baseline equals shared local policy value");
            }, player);
          }
          const double pre_update = row->Ev(root, *sampler, frozen);
          const double returned = row->UpdateAndReturnEv(root, *sampler, frozen);
          Close(returned, pre_update, "pre-update row payoff");
          SamePolicies(root, frozen, columns[col]->Frozen());
        }
        if (col == 0) {
          for (const auto& untouched : rows[1])
            Close(untouched->StageOriginalLocalRegret(), 0, "independent rows per opponent");
        }
        if (t == 0) {
          columns[col]->UpdateAndReturnEv(root, *sampler, columns[col]->Frozen());
          reference[col]->UpdateAndReturnEv(root, *sampler, reference[col]->Frozen());
          SamePolicies(root, columns[col]->Frozen(), reference[col]->Frozen());
        }
      }
    }
  }
}
}

int main() {
  try {
    hr_edl::Stopwatch<FakeClock> timer;
    FakeClock::elapsed = std::chrono::microseconds(500);
    Close(timer.milliseconds(), .5, "submillisecond timer");
    Close(timer.seconds(), .0005, "subsecond timer");
    RawCfrPlus();
    const auto selected = AlgsInGroup(11, 4);
    const std::vector<std::string> main_labels = {
        "CFR", "1-QB-EFR", "2-QB-EFR", "4-QB-EFR",
        "A-EFR_IN", "CSPS-EFR", "BEHAV-EFR"};
    if (AlgsInGroup(10, 4).size() != 14 || LearnerProfileLabels(selected) != main_labels)
      throw std::runtime_error("incorrect SCD main/legacy algorithm selection");
    FixedSemantics();
    for (size_t k : {0, 1, 2, 4}) for (size_t d : {0, 1, 2, 4, 5}) Coverage(d, k);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n'; return 1;
  }
  return 0;
}
