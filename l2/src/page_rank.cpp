#include "page_rank.h"

#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include <utility>

#include "../common/CycleTimer.h"
#include "../common/graph.h"


// pageRank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void pageRank(Graph g, double* solution, double damping, double convergence)
{


  // initialize vertex weights to uniform probability. Double
  // precision scores are used to avoid underflow for large graphs

  int numNodes = num_nodes(g);
  double equal_prob = 1.0 / numNodes;

  double* score_old = (double*) malloc(numNodes * sizeof(double));
  double* score_new = (double*) malloc(numNodes * sizeof(double));

  Vertex* sinkNodes = (Vertex*) malloc(numNodes * sizeof(double));
  int num_sink_nodes = 0;




  for (int i = 0; i < numNodes; ++i) {
    score_old[i] = equal_prob;
    if (outgoing_size(g, i) == 0) {
      sinkNodes[num_sink_nodes] = i;
      num_sink_nodes++;
    }
  }
  bool converged = false;
  
  /*
     CS149 students: Implement the page rank algorithm here.  You
     are expected to parallelize the algorithm using openMP.  Your
     solution may need to allocate (and free) temporary arrays.

     Basic page rank pseudocode is provided below to get you started:

     // initialization: see example code above
     score_old[vi] = 1/numNodes;

     while (!converged) {

       // compute score_new[vi] for all nodes vi:
       score_new[vi] = sum over all nodes vj reachable from incoming edges
                          { score_old[vj] / number of edges leaving vj  }
       score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / numNodes;

       score_new[vi] += sum over all nodes v in graph with no outgoing edges
                          { damping * score_old[v] / numNodes }

       // compute how much per-node scores have changed
       // quit once algorithm has converged

       global_diff = sum over all nodes vi { abs(score_new[vi] - score_old[vi]) };
       converged = (global_diff < convergence)
     }

   */
  while(!converged){     
	  double global_diff = 0.0;
	  int nthreads = omp_get_max_threads();
	  double local_sink_prob[nthreads];
    double local_diff[nthreads];
    for (int i = 0; i < nthreads; ++i) {
      local_sink_prob[i] = 0.0;
      local_diff[i] = 0.0;
    }
	
  
    // distribute the probability scores due to sink nodes
	  double sink_nodes_prob = 0.0;
	  
    #pragma omp parallel for 
    for (int i = 0; i < num_sink_nodes; ++i) {
      local_sink_prob[omp_get_thread_num()] += score_old[sinkNodes[i]] * damping / numNodes;
    }


	  for (int i = 0; i < nthreads; i++) {
      sink_nodes_prob += local_sink_prob[i];
    }

	  // compute new scores for all nodes
	  #pragma omp parallel for
	  for (int i = 0; i < numNodes; ++i) {
      score_new[i] = 0.0;
      int num_in_edges = incoming_size(g, i);
      const Vertex* varray = incoming_begin(g, i);

      for (int j = 0; j < num_in_edges; ++j) {
        int incoming_vertex = varray[j];
        score_new[i] += score_old[incoming_vertex] / outgoing_size(g, incoming_vertex);
      }

      // include damping
      score_new[i] = (damping * score_new[i]) + (1.0 - damping) / numNodes;
      
      // account for sink nodes i.e. nodes with no outgoing edges
      score_new[i] += sink_nodes_prob;
      local_diff[omp_get_thread_num()] += std::abs(score_new[i] - score_old[i]);
    }
	
	  for (int i = 0; i < nthreads; i++) {
      global_diff += local_diff[i];
    }
    
    converged = (global_diff < convergence);

    // swap new and old pagerank scores
    double* tmp = score_new;
    score_new = score_old;
    score_old = tmp;
  
  } // end while(!converged)

  for (int i = 0; i < numNodes; ++i) {
      solution[i] = score_old[i];
  }

  free(score_old);
  free(score_new);
  free(sinkNodes);
}
