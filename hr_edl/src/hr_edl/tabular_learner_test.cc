#include "hr_edl/adaptive_profile.h"

#include <iostream>
#include <vector>

namespace {

bool Check(const std::vector<double>& actual, const std::vector<double>& expected,
           const char* case_name) {
  if (actual == expected) return true;
  std::cerr << case_name << ": unexpected reach probabilities\n";
  return false;
}

}  // namespace

int main() {
  // Exact-k, noncausal deviations must skip exhausted memories when selecting
  // the most queried memory that can still query at this information set.
  const hr_edl::SCDPredecessors k4(4, false, false);
  if (!Check(k4.InternalPredecessorReachProbs({0.2, 0.3, 0.5}, {0, 2, 4}),
             {0.3}, "noncausal mixed budgets")) return 1;
  if (!Check(k4.InternalPredecessorReachProbs({0.2, 0.8}, {0, 4}),
             {0.8}, "noncausal no remaining-one memory")) return 1;
  if (!Check(k4.InternalPredecessorReachProbs({0.2, 0.8}, {0, 0}),
             {}, "noncausal exhausted")) return 1;

  // The previous and updated strategies must filter the same memory indices.
  const std::vector<size_t> remaining = {0, 2, 4};
  if (!Check(k4.InternalPredecessorReachProbs({0.2, 0.3, 0.5}, remaining),
             {0.3}, "previous strategy")) return 1;
  if (!Check(k4.InternalPredecessorReachProbs({0.1, 0.6, 0.3}, remaining),
             {0.6}, "updated strategy")) return 1;

  const hr_edl::SCDPredecessors causal(2, false, true);
  const double strategy[] = {0.25, 0.75};
  const auto [probabilities, budgets] = causal.SuccessorReachProbs(
      {1.0}, strategy, 2, 0, {2});
  if (!Check(probabilities, {1.0, 0.75, 0.25}, "causal successor")) return 1;
  if (budgets != std::vector<size_t>({2, 0, 1})) return 1;
  if (!Check(causal.ExternalPredecessorReachProbs(probabilities, budgets),
             {0.75}, "causal external")) return 1;
  if (!Check(causal.InternalPredecessorReachProbs(probabilities, budgets),
             {1.0, 0.25}, "causal internal")) return 1;
  return 0;
}
