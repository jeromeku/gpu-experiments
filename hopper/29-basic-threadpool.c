/*
    Conclusion: 
    - Using threadpool pattern, the overhead of dispatching a single task becomes
      0.2 ms -> 0.04 ms
    - The construction and destruction of threadpool combined is 0.4 ms

    (ignore numbers in the comments below; run again to be sure)
*/

#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>

#include <iostream> // for testing
#include <chrono> // for testing

class ThreadGang {
public:
    ThreadGang(size_t num_threads);
    ~ThreadGang();

    void execute(std::function<void(int)> task); // also waits for all threads to finish

private:
    // Condition indicators
    bool stop;
    std::vector<bool> task_available;
    int n_task_done;

    // Threadpool
    std::vector<std::thread> workers;
    
    // Main entry point for each thread
    void worker(int thread_id);

    // Used to dispatch work to all threads
    std::function<void(int)> current_task;

    // Synchronization
    std::mutex mutex;
    std::condition_variable cond_task_available;
    std::condition_variable cond_task_done;
};

ThreadGang::ThreadGang(size_t num_threads) : stop(false), n_task_done(0) {
    for (size_t i = 0; i < num_threads; ++i) {
        task_available.push_back(false);
        workers.emplace_back([this, i] { worker(i); });
    }
}

ThreadGang::~ThreadGang() {
    {
        std::lock_guard<std::mutex> lock(mutex);
        stop = true;
    }
    cond_task_available.notify_all();
    for (std::thread &worker : workers) {
        worker.join();
    }
}

void ThreadGang::execute(std::function<void(int)> task) {
    {
        std::lock_guard<std::mutex> lock(mutex);
        current_task = task;
        for (size_t i = 0; i < task_available.size(); ++i)
            task_available[i] = true;
    }
    cond_task_available.notify_all();
    {
        std::unique_lock<std::mutex> lock(mutex);
        cond_task_done.wait(lock, [this] { return n_task_done == workers.size(); });
        n_task_done = 0;
    }
}

void ThreadGang::worker(int thread_id) {
    while (true) {
        std::function<void(int)> task;
        {
            std::unique_lock<std::mutex> lock(mutex);
            cond_task_available.wait(lock, [this, thread_id] { return stop || task_available[thread_id]; });

            if (stop)
                return;

            task = current_task;
            task_available[thread_id] = false;
        }
        task(thread_id);
        {
            std::lock_guard<std::mutex> lock(mutex); // adds 10 microseconds overhead
            ++n_task_done;
            if (n_task_done == workers.size())
                cond_task_done.notify_one();
        }
    }
}

#define NUM_THREADS 8

int main() {
    auto start = std::chrono::high_resolution_clock::now();

    ThreadGang *gang = new ThreadGang(NUM_THREADS); // 0.23 ms

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Constructor execution time: " << elapsed.count() * 1e3 << " ms." << std::endl;

    // Must pass in a function that takes an int (thread ID) as an argument
    
    start = std::chrono::high_resolution_clock::now();

    // 0.041353 ms for purely overhead of running task
    gang->execute([](int thread_id) {
        // std::cout << "Task" << std::endl;
    });

    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "Tast execution time 1: " << elapsed.count() * 1e3 << " ms." << std::endl;

    start = std::chrono::high_resolution_clock::now();

    // 0.005719 ms 
    gang->execute([](int thread_id) {
        std::cout << "Task" << std::endl;
    });

    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "Task execution time 2: " << elapsed.count() * 1e3 << " ms." << std::endl;

    start = std::chrono::high_resolution_clock::now();

    delete gang;

    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "Threadpool destruction time: " << elapsed.count() * 1e3 << " ms." << std::endl;

    {
        std::vector<std::thread> threads;

        start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < NUM_THREADS; ++i) {
            threads.push_back(std::thread([i] {
                std::cout << "Task" << i << std::endl;
            }));
        }

        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        std::cout << "Raw threads execution time: " << elapsed.count() * 1e3 << " ms." << std::endl;
        
        for (int i = 0; i < NUM_THREADS; ++i) {
            threads[i].join();
        }
    }

    {
        std::vector<std::thread> threads;

        start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < NUM_THREADS; ++i) {
            threads.push_back(std::thread([i] {
                std::cout << "Task" << i << std::endl;
            }));
        }
        
        for (int i = 0; i < NUM_THREADS; ++i) {
            threads[i].join();
        }

        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        std::cout << "Raw threads execution + join time: " << elapsed.count() * 1e3 << " ms." << std::endl;
    }

    {
        std::vector<std::thread> threads;

        for (int i = 0; i < NUM_THREADS; ++i) {
            threads.push_back(std::thread([i] {
                std::cout << "Task" << i << std::endl;
            }));
        }

        start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < NUM_THREADS; ++i) {
            threads[i].join();
        }

        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        std::cout << "Raw threads join time: " << elapsed.count() * 1e3 << " ms." << std::endl;
    }

    std::this_thread::sleep_for(std::chrono::seconds(2));

    return 0;
}
