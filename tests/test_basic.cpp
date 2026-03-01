#include <cassert>
#include <cstdio>
#include <filesystem>
#include <vector>

#include "vecgraphdb/vecgraphdb.hpp"

int main() {
  namespace fs = std::filesystem;
  fs::path tmp = fs::temp_directory_path() / "vecgraphdb_test";
  fs::remove_all(tmp);

  vecgraphdb::VecGraphDB db(tmp.string(), 3);

  db.add_pair("a", {1,2,3}, "b", {1,2,2.9f});
  db.add_vector("c", {0,0,1});
  db.flush();

  auto sim = db.topk_similar({1,2,3}, 2);
  assert(!sim.empty());

  auto corr = db.topk_correlated({1,2,3}, 2);
  // at least b should be correlated to a (via nearest anchor)
  assert(!corr.empty());

  auto corr2 = db.topk_correlated_by_id("a", 2);
  assert(!corr2.empty());

  std::printf("OK\n");
  return 0;
}
