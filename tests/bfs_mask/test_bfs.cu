// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

// Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.

/**
 * @file
 * test_bfs.cu
 *
 * @brief Simple test driver program for breadth-first search.
 */

#include <stdio.h>
#include <string>
#include <deque>
#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>

// Utilities and correctness-checking
#include <gunrock/util/test_utils.cuh>
#include <gunrock/util/track_utils.cuh>

// BFS includes
#include <gunrock/app/bfs/bfs_enactor.cuh>
#include <gunrock/app/bfs/bfs_problem.cuh>
#include <gunrock/app/bfs/bfs_functor.cuh>

// Operator includes
#include <gunrock/oprtr/advance/kernel.cuh>
#include <gunrock/oprtr/filter/kernel.cuh>

#include <moderngpu.cuh>

using namespace gunrock;
using namespace gunrock::app;
using namespace gunrock::util;
using namespace gunrock::oprtr;
using namespace gunrock::app::bfs;

void ref_bfs_mask(const int src_node, const int dst_node, const int num_nodes, const int num_edges, const int *row_offsets, const int *col_indices, const int *col_mask, int *parents)
{
  int *q = (int*)malloc(num_nodes * sizeof(int));
  q[0] = src_node;
  parents[src_node] = src_node;
  int idx = -1;
  int size = 1;
  int found = 0;
  while (idx+1 < size && !found) {
    idx++;
    int u = q[idx];
    for (int i = row_offsets[u]; i < row_offsets[u+1]; i++) {
      int v = col_indices[i];
      if (parents[v] == -1 && col_mask[i]) {
        parents[v] = u;
        if (v == dst_node) {
          found = 1;
          break;
        }
        else {
          q[size] = v;
          size++;
        }
      }
    }
  }
}

void bfs_mask(const int src_node, const int dst_node, const int num_nodes, const int num_edges, const int *row_offsets, const int *col_indices, const int *col_mask, int *parents)
{
  // TODO: use Gunrock's customized BFS here
  ref_bfs_mask(src_node, dst_node, num_nodes, num_edges, row_offsets, col_indices, col_mask, parents);
}

int main(int argc, char** argv)
{
  // initialize graph here
  int num_nodes = 7, num_edges = 14, src_node = 0, dst_node = 6;
  int row_offsets[8]  = {0, 2, 5, 8, 10, 13, 14, 14};
  int col_indices[14] = {1, 2, 0, 2, 4, 3, 4, 5, 5, 6, 2, 5, 6, 6};
  int col_mask[14]    = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

  // setup memory in gpu
  int *d_row_offsets;
  cudaMallocManaged(&d_row_offsets, (num_nodes + 1) * sizeof(int));
  memcpy(d_row_offsets, row_offsets, (num_nodes + 1) * sizeof(int));
  int *d_col_indices;
  cudaMallocManaged(&d_col_indices, num_edges * sizeof(int));
  memcpy(d_col_indices, col_indices, num_edges * sizeof(int));
  int *d_col_mask;
  cudaMallocManaged(&d_col_mask, num_edges * sizeof(int));
  memcpy(d_col_mask, col_mask, num_edges * sizeof(int));
  int *d_parents;
  cudaMallocManaged(&d_parents, num_nodes * sizeof(int));
  for (int node = 0; node < num_nodes; node++) 
    d_parents[node] = -1;
 
  // run bfs (with mask)
  bfs_mask(src_node, dst_node, num_nodes, num_edges, d_row_offsets, d_col_indices, d_col_mask, d_parents);

  // print out results
  for (int node = 0; node < num_nodes; node++)
    printf("Node_ID [%d] : Parent [%d]\n", node, d_parents[node]);

  // free memory
  cudaFree(d_row_offsets);
  cudaFree(d_col_indices);
  cudaFree(d_col_mask);
  cudaFree(d_parents);

  return 0;
}
// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
