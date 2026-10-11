#include <limits>
#include <memory>
#include <optional>
#include <queue>
#include <string>
#include <tuple>
#include <vector>

#include <fstream>
#include <filesystem>


#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "bin/ltbr.h"
#include "eigen/Eigen/Dense"
#include "hr_edl/adaptive_profile.h"
#include "hr_edl/best_response.h"
#include "hr_edl/decision_point.h"
#include "hr_edl/policy_evaluation.h"
#include "hr_edl/spiel_extra.h"
#include "hr_edl/stopwatch.h"
#include "open_spiel/game_transforms/turn_based_simultaneous_game.h"
#include "open_spiel/spiel.h"

// Environment
ABSL_FLAG(std::string, game, "leduc_poker", "The game to play.");
ABSL_FLAG(std::string, sampler, "null", "The sampler to use.");
ABSL_FLAG(size_t, t, 1000, "The number of iterations to run.");
ABSL_FLAG(size_t, random_seed, 0, "Seed for a sampler's random engine.");

// Modes
ABSL_FLAG(bool, show_num, false,
          "Show the number of table entries that would be produced without "
          "computing them.");
ABSL_FLAG(size_t, alg_group, 0,
          "0 => efficient EFR instances only, 1 => efficient and behavioral "
          "deviation EFR instances.");

// Parallelism
ABSL_FLAG(size_t, threads, 1, "The number of threads to use.");

// 轮流计算regret，但不同于RM中self-play的轮流策略更新方式
// 这里一定包含不同于自己的对手，不属于self-play，弃用
ABSL_FLAG(bool, alt, false,
          "Whether to use the alternative way to compute the regrets.");
ABSL_FLAG(bool, cfr_plus, true,
          "Apply RM+ truncation to cumulative solver regrets.");

// RT parameters
ABSL_FLAG(double, rt_cfr_weight, 0.0, "The regularization weight for the rt policy update in RT-CFR.");
ABSL_FLAG(size_t, rt_cfr_T, 1, "The period for the rt policy update in RT-CFR.");


// Perturbation parameter
ABSL_FLAG(double, epsilon, 0.0, "The perturbation parameter for perturbed LTBR.");

// file directory
ABSL_FLAG(std::string, file_dir, "data/test_results.ssv",
          "The directory to store the results.");

// qbce
ABSL_FLAG(size_t, query_budget, 0, "The number of queries for k-QBCE.");

void run_experiment() {
  // 获取游戏id
  const std::string game_name = absl::GetFlag(FLAGS_game);
  // 根据游戏id加载游戏，直接读取游戏或者加载efg格式文件游戏
  std::shared_ptr<const open_spiel::Game> game;
  if (absl::StrContains(game_name, ".efg")) {
    std::cout << "Loading game from file: " << game_name << std::endl;
    if (!std::filesystem::exists(game_name)) {
      std::cerr << "Error: File does not exist: " << game_name << std::endl;
      return;
    }
    // 从efg文件加载游戏
    game = open_spiel::LoadGameAsTurnBased(
          "efg_game", {{"filename", open_spiel::GameParameter(
                                        game_name)}});
    if (!game) {
      std::cerr << "Error: Failed to load game from file: " << game_name
                << std::endl;
      return;
    }
  } else {
    // 从游戏id加载游戏
    game = open_spiel::LoadGameAsTurnBased(game_name);
    if (!game) {
      std::cerr << "Error: Failed to load game: " << game_name << std::endl;
      return;
    }
  }

  const bool show_num = absl::GetFlag(FLAGS_show_num);
  // 获取算法组，0-9，每个数字对应一组算法，具体算法见 AlgsInGroup() 函数

  const std::unique_ptr<open_spiel::State> root_state = game->NewInitialState();

  const int random_seed = absl::GetFlag(FLAGS_random_seed);
  const size_t iterations = absl::GetFlag(FLAGS_t);

  const int num_threads = absl::GetFlag(FLAGS_threads);
  //获取文件目录 e.g. data/test_results.ssv
  const std::string file_dir = absl::GetFlag(FLAGS_file_dir);

  const std::string sampler_name = absl::GetFlag(FLAGS_sampler);
  std::unique_ptr<hr_edl::MccfrSampler> sampler =
      hr_edl::NewSampler(sampler_name, random_seed);

  const double rt_cfr_weight = absl::GetFlag(FLAGS_rt_cfr_weight);
  const size_t rt_cfr_T = absl::GetFlag(FLAGS_rt_cfr_T);

  const double epsilon = absl::GetFlag(FLAGS_epsilon);

  //获取 game_length, 例如 Kuhn poker 3p 的 game_length 是 4，Leduc poker 2p 的 game_length 是 6，Leduc poker 3p 的 game_length 是 8
  const size_t game_length = game->MaxGameLength();
  // 获取游戏玩家shuliang
  const int num_players = game->NumPlayers();
  //计算游戏深度, 向上取整
  const size_t game_depth = (game_length + num_players - 1) / num_players;
  const size_t alg_group = absl::GetFlag(FLAGS_alg_group); // 1
     // 获取算法组，0表示只测试高效算法，1表示测试高效算法和行为偏差算法，
     // 2表示测试rt-cfr算法，3表示测试扰动ltbr算法，4表示测试qbce算法，5表示测试qbce_full算法，
     // 6表示测试不同k-的qbce算法，7表示测试不同k-的qbce-seq算法，
     // 8表示测试不同k-的qbce-cs算法，9表示测试不同k-的qbce-seq-cs算法
  // 如果是qbce单一算法测试组 {6,7,8,9}, 使用game_depth作为k值传入算法构造函数， 否则使用FLAGS_query_budget作为k值传入算法构造函数
  // 10:  表示单独测试infty-QB-EFR、CFR(0-QB-EFR)、1-QB-EFR, CSPS, BHV算法
  size_t k = (alg_group >= 6 && alg_group <= 9) ? game_depth : absl::GetFlag(FLAGS_query_budget);
  const auto labeled_algs = AlgsInGroup(alg_group, k,
   absl::GetFlag(FLAGS_rt_cfr_weight), absl::GetFlag(FLAGS_rt_cfr_T), absl::GetFlag(FLAGS_epsilon));


  const auto alg_labels = LearnerProfileLabels(labeled_algs);
  if (show_num) {
    std::cout << alg_labels.size() * (alg_labels.size() + 1) << std::endl;
    return;
  }

// 存储结果的矩阵，行数是算法数量+1（多一行存储BR的结果），列数是迭代次数
  std::vector<Eigen::MatrixXd> expected_values;

  std::cout << "learner/env  ";
  for (const auto& label : alg_labels) {
    std::cout << label << "  ";
    expected_values.push_back(
        Eigen::MatrixXd::Zero(alg_labels.size(), iterations));
  }
  expected_values.push_back(
      Eigen::MatrixXd::Zero(alg_labels.size(), iterations));  // For BR row
  std::cout << std::endl;

  // 存储每次迭代的运行时间，行数是算法数量+1（多一行存储BR的结果），列数是迭代次数
  std::vector<Eigen::MatrixXd> milliseconds;
  for (const auto& label : alg_labels) {
    milliseconds.push_back(
        Eigen::MatrixXd::Zero(alg_labels.size(), iterations));
  }
  milliseconds.push_back(
      Eigen::MatrixXd::Zero(alg_labels.size(), iterations));  // For BR row

  // Maximum maintained local solver regret. This is not protocol-level OSR.
  // BR has no cumulative solver regrets, so its metric is unavailable.
  std::vector<Eigen::MatrixXd> max_full_phi_regrets_fus;
  for (size_t row_alg = 0; row_alg < alg_labels.size(); ++row_alg) {
    max_full_phi_regrets_fus.push_back(
        Eigen::MatrixXd::Zero(alg_labels.size(), iterations));
  }
  max_full_phi_regrets_fus.push_back(Eigen::MatrixXd::Constant(
      alg_labels.size(), iterations, std::numeric_limits<double>::quiet_NaN()));
  // Stage original accumulators start at zero and are never reset here.
  // They provide untruncated retained local regrets even with CFR+.
  std::vector<Eigen::MatrixXd> retained_local_regrets;
  for (size_t row_alg = 0; row_alg < alg_labels.size() + 1; ++row_alg) {
    retained_local_regrets.push_back(Eigen::MatrixXd::Constant(
        alg_labels.size(), iterations, std::numeric_limits<double>::quiet_NaN()));
  }
  auto diagnostic_milliseconds = retained_local_regrets;

  const double utility_diameter = game->MaxUtility() - game->MinUtility();
  // 为每个列算法创建一个学习器配置文件，行算法使用这些配置文件进行更新，列算法也使用这些配置文件进行更新，
  // 每个配置文件对应一个算法，配置文件的数量等于列算法的数量
  std::vector<hr_edl::AdaptiveProfilePtr> col_learner_profiles =
      NewLearnerProfiles(labeled_algs, game->NumPlayers(), utility_diameter);
  const bool cfr_plus = absl::GetFlag(FLAGS_cfr_plus);
  for (auto& learner : col_learner_profiles) learner->SetCfrPlus(cfr_plus);

  std::queue<std::thread> threads;

  // 创建根节点原件
  hr_edl::CachedDecisionPoint root(root_state->Clone(), false, true);
  for (size_t col_alg = 0; col_alg < col_learner_profiles.size(); ++col_alg) {
    // 每个列算法在一个线程中运行，行算法在同一线程中运行
    const auto f = [&milliseconds, &expected_values, &max_full_phi_regrets_fus,
                    &retained_local_regrets,
                    &diagnostic_milliseconds,
                    &col_learner_profiles, &root_template = root, &sampler,
                    &labeled_algs, col_alg, iterations, utility_diameter, rt_cfr_weight, rt_cfr_T, epsilon, cfr_plus]() {
      hr_edl::Stopwatch stop_watch;
      // 每个线程都有一个独立的根节点和行算法配置文件，避免线程之间的竞争
      hr_edl::CachedDecisionPoint root(root_template);
      std::vector<hr_edl::AdaptiveProfilePtr> row_algs;
      // 行算法的数量比列算法多一个，因为行算法还包括一个BR算法，
      // BR算法的配置文件在这里创建，使用BestResponseProfile类实现
      for (size_t row_alg = 0; row_alg < col_learner_profiles.size() + 1;
           ++row_alg) {
        // 前n个使用对应的算法创建配置文件，最后一个使用BR算法创建配置文件
        row_algs.emplace_back(
            row_alg < col_learner_profiles.size()
                ? labeled_algs[row_alg].New(root.NumPlayers(), utility_diameter)
                : hr_edl::AdaptiveProfilePtr(
                      new hr_edl::BestResponseProfile(root.NumPlayers())));
        row_algs.back()->SetCfrPlus(cfr_plus);
      }
      for (size_t t = 0; t < iterations; ++t) {
          // debug 打印当前迭代次数
          std::cout << "Iteration " << t
          << " of row algorithms vs " << labeled_algs[col_alg].label_ << std::endl;
        // 固定对手列算法
        const auto col_profile = col_learner_profiles[col_alg]->Frozen();
        // 使用不同的行算法进行博弈
        for (size_t row_alg = 0; row_alg < col_learner_profiles.size() + 1;
             ++row_alg) {
          // 计算row_alg本轮博弈的收益以及耗时，其中针对对手的策略更新自己策略
          stop_watch.reset();
          const auto v = row_algs[row_alg]->UpdateAndReturnEv(
              root, *sampler, col_profile);
          milliseconds[row_alg](col_alg, t) = stop_watch.milliseconds();
          expected_values[row_alg](col_alg, t) = v;
          if (row_alg < col_learner_profiles.size()) {
            stop_watch.reset();
            const auto regrets = row_algs[row_alg]->PhiRegrets();
            max_full_phi_regrets_fus[row_alg](col_alg, t) = std::get<0>(regrets);
            retained_local_regrets[row_alg](col_alg, t) =
                row_algs[row_alg]->StageOriginalLocalRegret();
            diagnostic_milliseconds[row_alg](col_alg, t) = stop_watch.milliseconds();
          }
        }
        stop_watch.reset();
        // col_alg self-play, 使用自己的策略更新自己的策略,
        // 这里区别 sim 更新, fixed 下对手的策略更新是单独进行
        if (t < iterations - 1) {
          col_learner_profiles[col_alg]->UpdateAndReturnEv(
              root, *sampler, col_learner_profiles[col_alg]->Frozen());
        }
      }
    };
    if (num_threads > 1) {
      threads.emplace(f);

      if (threads.size() >= num_threads) {
        hr_edl::Wait(threads.front());
        threads.pop();
      }
    } else {
      f();
    }
  }
  hr_edl::Wait(threads);

  // 输出结果到指定文件
  // 添加作用域，确保文件流在恢复 cout 之前关闭
  namespace fs = std::filesystem;

    // 1. 确保目录存在
    try {
        fs::path path(file_dir);
        if (!path.parent_path().empty()) {
            fs::create_directories(path.parent_path());
        }
    } catch (const std::exception& e) {
        std::cerr << "创建目录失败: " << e.what() << std::endl;
        return;
    }

    // 2. 打开文件（覆盖模式）
    std::ofstream ofs(file_dir, std::ios::trunc); // 或追加模式 std::ios::app
    if (!ofs.is_open()) {
        std::cerr << "无法打开文件: " << file_dir << std::endl;
        return;
    }

    // 3. 保存并重定向 cout
    std::streambuf* cout_buf = std::cout.rdbuf();
    std::cout.rdbuf(ofs.rdbuf());

    // 4. RAII 恢复器（关键！防止异常导致 cout 永久失效）
    struct RaiiRestore {
        std::streambuf* buf;
        RaiiRestore(std::streambuf* b) : buf(b) {}
        ~RaiiRestore() { std::cout.rdbuf(buf); }
    } raii(cout_buf);

    // ------------------- 原始打印逻辑（保持不变） -------------------
    // print table header
    std::cout << "# Each entry: (expected_value, milliseconds, max_full_phi_regret_fus, retained_local_regret_numerator, diagnostic_milliseconds)\n";
    std::cout << "# t is zero-based; each row learner has t+1 updates; value uses its pre-update policy.\n";
    std::cout << "# cfr_plus = " << (cfr_plus ? "true" : "false") << '\n';
    std::cout << "# metric_evaluation_set = maintained_local_EX_IN_experts; maximized over seats\n";
    std::cout << "learner/env  ";
    for (const auto& label : alg_labels) {
        std::cout << label << "  ";
    }
    std::cout << '\n';
  for (size_t t = 0; t < iterations; ++t) {
    std::cout << "t = " << t << std::endl;
    for (size_t row_alg = 0; row_alg < col_learner_profiles.size() + 1;
         ++row_alg) {
      // 获取row label 算法标签或者 "BR"
      const std::string row_label =
          row_alg < col_learner_profiles.size() ? alg_labels[row_alg] : "BR";
      std::cout << row_label << "  ";
      for (size_t col_alg = 0; col_alg < col_learner_profiles.size();
           ++col_alg) {
        std::cout << absl::StrFormat("(%.17g, %.17g, %.17g, %.17g, %.17g)  ",
                                     expected_values[row_alg](col_alg, t),
                                     milliseconds[row_alg](col_alg, t),
                                     max_full_phi_regrets_fus[row_alg](col_alg, t),
                                     retained_local_regrets[row_alg](col_alg, t),
                                     diagnostic_milliseconds[row_alg](col_alg, t));
      }
      std::cout << std::endl;
    }
  }
    // ---------------------------------------------------------------

    // 5. 刷新确保写入磁盘（可选）
    ofs.flush();

    // 6. RAII 自动恢复 cout（raii 析构）
    // restore_cout();  // 可省略，raii 会做
  return;
}

int main(int argc, char** argv) {
  absl::SetProgramUsageMessage(
      "Run a learning tournament with fixed compatriots.");
  absl::ParseCommandLine(argc, argv);
  run_experiment();
}
