#ifndef HR_EDL_STOPWATCH_H_
#define HR_EDL_STOPWATCH_H_

#include <atomic>
#include <chrono>

namespace hr_edl {

// Thanks to Incomputable
// https://codereview.stackexchange.com/questions/196245/extremely-simple-timer-class-in-c
// for this.
template <typename Clock = std::chrono::steady_clock>
class Stopwatch {
 public:
  Stopwatch() : start_time_(Clock::now()) {} //构造函数，初始化开始时间为当前时间

  void reset() { start_time_ = Clock::now(); } //重置计时器，重新开始计时

  template <typename Rep = typename Clock::duration::rep,
            typename Units = typename Clock::duration> //模板函数，返回指定单位的时间
  Rep duration() const {  
    std::atomic_thread_fence(std::memory_order_relaxed);
    const auto duration =
        std::chrono::duration_cast<Units>(Clock::now() - start_time_).count();
    std::atomic_thread_fence(std::memory_order_relaxed);
    return static_cast<Rep>(duration);
  }

  double seconds() const { return duration<double, std::chrono::duration<double>>(); } //返回秒数
  double milliseconds() const { //返回毫秒数
    return duration<double, std::chrono::duration<double, std::milli>>();
  }

 private:
  typename Clock::time_point start_time_;
};
}  // namespace hr_edl

#endif  // HR_EDL_STOPWATCH_H_
