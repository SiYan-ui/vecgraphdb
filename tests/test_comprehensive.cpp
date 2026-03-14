#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <filesystem>
#include <vector>
#include <random>
#include <thread>
#include <atomic>

#include "vecgraphdb/vecgraphdb.hpp"

namespace fs = std::filesystem;

// Test helper to generate random vectors
static std::vector<float> random_vector(std::size_t dim, std::mt19937& gen) {
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> v(dim);
  for (std::size_t i = 0; i < dim; ++i) {
    v[i] = dist(gen);
  }
  return v;
}

TEST_CASE("Basic construction and insertion") {
  fs::path tmp = fs::temp_directory_path() / "vecgraphdb_test_basic";
  fs::remove_all(tmp);
  
  vecgraphdb::VecGraphDB db(tmp.string(), 3, "cosine");
  
  db.add_vector("a", {1.0f, 0.0f, 0.0f});
  db.add_vector("b", {0.0f, 1.0f, 0.0f});
  db.add_vector("c", {0.0f, 0.0f, 1.0f});
  
  auto stats = db.get_stats();
  CHECK(stats.count == 3);
  CHECK(stats.dim == 3);
  CHECK(stats.vectors_bytes > 0);
}

TEST_CASE("Duplicate ID rejection") {
  fs::path tmp = fs::temp_directory_path() / "vecgraphdb_test_dup";
  fs::remove_all(tmp);
  
  vecgraphdb::VecGraphDB db(tmp.string(), 3);
  db.add_vector("id1", {1, 2, 3});
  
  CHECK_THROWS_AS(db.add_vector("id1", {4, 5, 6}), std::invalid_argument);
}

TEST_CASE("Batch insertion") {
  fs::path tmp = fs::temp_directory_path() / "vecgraphdb_test_batch";
  fs::remove_all(tmp);
  
  vecgraphdb::VecGraphDB db(tmp.string(), 3);
  
  std::vector<std::string> ids = {"a", "b", "c", "d"};
  std::vector<std::vector<float>> vecs = {
    {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 1, 0}
  };
  
  db.add_vectors_batch(ids, vecs);
  
  auto stats = db.get_stats();
  CHECK(stats.count == 4);
}

TEST_CASE("Batch insertion with duplicate check") {
  fs::path tmp = fs::temp_directory_path() / "vecgraphdb_test_batch_dup";
  fs::remove_all(tmp);
  
  vecgraphdb::VecGraphDB db(tmp.string(), 3);
  db.add_vector("existing", {1, 2, 3});
  
  std::vector<std::string> ids = {"new1", "existing"};
  std::vector<std::vector<float>> vecs = {{1, 0, 0}, {0, 1, 0}};
  
  CHECK_THROWS_AS(db.add_vectors_batch(ids, vecs), std::invalid_argument);
}

TEST_CASE("Pair insertion with correlation") {
  fs::path tmp = fs::temp_directory_path() / "vecgraphdb_test_pair";
  fs::remove_all(tmp);
  
  vecgraphdb::VecGraphDB db(tmp.string(), 3);
  db.add_pair("a", {1, 2, 3}, "b", {1, 2, 2.9f}, 0.95f);
  
  auto corr = db.topk_correlated_by_id("a", 5);
  CHECK(corr.size() == 1);
  CHECK(corr[0].id == "b");
  CHECK(corr[0].corr == 0.95f);
}

TEST_CASE("Similarity search") {
  fs::path tmp = fs::temp_directory_path() / "vecgraphdb_test_similar";
  fs::remove_all(tmp);
  
  vecgraphdb::VecGraphDB db(tmp.string(), 3);
  db.add_vector("a", {1, 0, 0});
  db.add_vector("b", {0.9f, 0.1f, 0});
  db.add_vector("c", {0, 1, 0});
  
  auto results = db.topk_similar({1, 0, 0}, 2);
  CHECK(results.size() == 2);
  CHECK(results[0].id == "a"); // Most similar to itself
  CHECK(results[0].score > 0.99f);
}

TEST_CASE("Correlated query by vector") {
  fs::path tmp = fs::temp_directory_path() / "vecgraphdb_test_corr_vec";
  fs::remove_all(tmp);
  
  vecgraphdb::VecGraphDB db(tmp.string(), 3);
  db.add_pair("a", {1, 2, 3}, "b", {1, 2, 2.9f});
  db.add_vector("c", {0, 0, 1});
  
  auto results = db.topk_correlated({1, 2, 3}, 5);
  CHECK(!results.empty());
}

TEST_CASE("Correlated query by ID") {
  fs::path tmp = fs::temp_directory_path() / "vecgraphdb_test_corr_id";
  fs::remove_all(tmp);
  
  vecgraphdb::VecGraphDB db(tmp.string(), 3);
  db.add_pair("a", {1, 2, 3}, "b", {1, 2, 2.9f});
  db.add_pair("a", {1, 2, 3}, "c", {0.9f, 1.8f, 2.7f}, 0.8f);
  
  auto results = db.topk_correlated_by_id("a", 5);
  CHECK(results.size() == 2);
  // Should be sorted by correlation descending
  CHECK(results[0].corr >= results[1].corr);
}

TEST_CASE("Unknown ID in correlated query") {
  fs::path tmp = fs::temp_directory_path() / "vecgraphdb_test_unknown";
  fs::remove_all(tmp);
  
  vecgraphdb::VecGraphDB db(tmp.string(), 3);
  db.add_vector("a", {1, 2, 3});
  
  CHECK_THROWS_AS(db.topk_correlated_by_id("nonexistent", 5), std::invalid_argument);
}

TEST_CASE("Persistence and reload") {
  fs::path tmp = fs::temp_directory_path() / "vecgraphdb_test_persist";
  fs::remove_all(tmp);
  
  // Create and populate database
  {
    vecgraphdb::VecGraphDB db(tmp.string(), 3);
    db.add_pair("a", {1, 2, 3}, "b", {1, 2, 2.9f});
    db.add_vector("c", {0, 0, 1});
    db.flush();
  }
  
  // Reload and verify
  {
    vecgraphdb::VecGraphDB db(tmp.string(), 3);
    auto stats = db.get_stats();
    CHECK(stats.count == 3);
    
    auto results = db.topk_similar({1, 2, 3}, 5);
    CHECK(!results.empty());
    CHECK(results[0].id == "a");
    
    auto corr = db.topk_correlated_by_id("a", 5);
    CHECK(!corr.empty());
  }
}

TEST_CASE("HNSW parameter tuning") {
  fs::path tmp = fs::temp_directory_path() / "vecgraphdb_test_hnsw";
  fs::remove_all(tmp);
  
  vecgraphdb::VecGraphDB db(tmp.string(), 3, "cosine", 8, 50, 30);
  db.add_vector("a", {1, 0, 0});
  db.add_vector("b", {0, 1, 0});
  
  db.set_hnsw_ef_search(100);
  db.reserve(1000);
  
  auto stats = db.get_stats();
  CHECK(stats.count == 2);
}

TEST_CASE("Empty database queries") {
  fs::path tmp = fs::temp_directory_path() / "vecgraphdb_test_empty";
  fs::remove_all(tmp);
  
  vecgraphdb::VecGraphDB db(tmp.string(), 3);
  
  auto similar = db.topk_similar({1, 2, 3}, 5);
  CHECK(similar.empty());
  
  auto corr = db.topk_correlated({1, 2, 3}, 5);
  CHECK(corr.empty());
}

TEST_CASE("Dimension mismatch error") {
  fs::path tmp = fs::temp_directory_path() / "vecgraphdb_test_dim";
  fs::remove_all(tmp);
  
  vecgraphdb::VecGraphDB db(tmp.string(), 3);
  
  CHECK_THROWS_AS(db.add_vector("a", {1, 2}), std::invalid_argument);
  CHECK_THROWS_AS(db.topk_similar({1, 2}, 5), std::invalid_argument);
}

TEST_CASE("Concurrent reads (thread safety)") {
  fs::path tmp = fs::temp_directory_path() / "vecgraphdb_test_concurrent";
  fs::remove_all(tmp);
  
  // Setup database
  {
    vecgraphdb::VecGraphDB db(tmp.string(), 10);
    for (int i = 0; i < 100; ++i) {
      db.add_vector("id" + std::to_string(i), {1.0f, 0, 0, 0, 0, 0, 0, 0, 0, 0});
    }
    db.flush();
  }
  
  // Reopen for concurrent reads
  vecgraphdb::VecGraphDB db(tmp.string(), 10);
  
  std::atomic<int> success_count{0};
  std::vector<std::thread> threads;
  
  // Launch multiple reader threads
  for (int t = 0; t < 4; ++t) {
    threads.emplace_back([&db, &success_count]() {
      for (int i = 0; i < 50; ++i) {
        auto results = db.topk_similar({1.0f, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 5);
        if (!results.empty()) {
          success_count.fetch_add(1);
        }
      }
    });
  }
  
  for (auto& thread : threads) {
    thread.join();
  }
  
  CHECK(success_count.load() == 200); // 4 threads * 50 iterations
}

TEST_CASE("MANIFEST.json creation and validation") {
  fs::path tmp = fs::temp_directory_path() / "vecgraphdb_test_manifest";
  fs::remove_all(tmp);
  
  // Create database
  {
    vecgraphdb::VecGraphDB db(tmp.string(), 5);
    db.add_vector("a", {1, 2, 3, 4, 5});
    db.flush();
  }
  
  // Check manifest exists
  fs::path manifest_path = tmp / "data" / "MANIFEST.json";
  CHECK(fs::exists(manifest_path));
  
  // Verify dimension mismatch detection
  CHECK_THROWS_AS(vecgraphdb::VecGraphDB(tmp.string(), 3), std::runtime_error);
}

TEST_CASE("Edge cases: k=0 queries") {
  fs::path tmp = fs::temp_directory_path() / "vecgraphdb_test_k0";
  fs::remove_all(tmp);
  
  vecgraphdb::VecGraphDB db(tmp.string(), 3);
  db.add_vector("a", {1, 2, 3});
  
  CHECK(db.topk_similar({1, 2, 3}, 0).empty());
  CHECK(db.topk_correlated({1, 2, 3}, 0).empty());
  CHECK(db.topk_correlated_by_id("a", 0).empty());
}

TEST_CASE("Large batch insertion") {
  fs::path tmp = fs::temp_directory_path() / "vecgraphdb_test_large_batch";
  fs::remove_all(tmp);
  
  vecgraphdb::VecGraphDB db(tmp.string(), 64);
  
  std::vector<std::string> ids;
  std::vector<std::vector<float>> vecs;
  
  std::mt19937 gen(42);
  for (int i = 0; i < 500; ++i) {
    ids.push_back("id" + std::to_string(i));
    vecs.push_back(random_vector(64, gen));
  }
  
  db.add_vectors_batch(ids, vecs);
  
  auto stats = db.get_stats();
  CHECK(stats.count == 500);
  
  // Verify we can query
  auto results = db.topk_similar(random_vector(64, gen), 10);
  CHECK(results.size() == 10);
}

TEST_CASE("Pearson correlation computation") {
  fs::path tmp = fs::temp_directory_path() / "vecgraphdb_test_pearson";
  fs::remove_all(tmp);
  
  vecgraphdb::VecGraphDB db(tmp.string(), 4);
  
  // Perfect positive correlation
  db.add_pair("a", {1, 2, 3, 4}, "b", {2, 4, 6, 8});
  auto corr_ab = db.topk_correlated_by_id("a", 1);
  CHECK(corr_ab[0].corr > 0.99f);
  
  // Perfect negative correlation
  db.add_pair("c", {1, 2, 3, 4}, "d", {4, 3, 2, 1}, -0.99f);
  auto corr_cd = db.topk_correlated_by_id("c", 1);
  CHECK(corr_cd[0].corr < -0.9f);
}
