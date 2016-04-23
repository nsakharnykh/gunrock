// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * parmetis_partitioner.cuh
 *
 * @brief linkage to parmetis partitioner
 */

#pragma once

#ifdef METIS_FOUND
  #include <metis.h>
#endif

#ifdef PARMETIS_FOUND
  #include <parmetis.h>
#endif

#include <gunrock/app/partitioner_base.cuh>
#include <gunrock/util/error_utils.cuh>

namespace gunrock {
namespace app {
namespace parmetis {

template <
    typename VertexId,
    typename SizeT,
    typename Value/*,
    bool     ENABLE_BACKWARD = false,
    bool     KEEP_ORDER      = false,
    bool     KEEP_NODE_NUM   = false*/>
struct ParMetisPartitioner : PartitionerBase<VertexId,SizeT,Value/*,
    ENABLE_BACKWARD, KEEP_ORDER, KEEP_NODE_NUM*/>
{
    typedef PartitionerBase<VertexId, SizeT, Value> BasePartitioner;
    typedef Csr<VertexId,SizeT,Value> GraphT;

    // Members
    float *weitage;

    // Methods
    /*MetisPartitioner()
    {
        weitage=NULL;
    }*/

    ParMetisPartitioner(
        const  GraphT &graph,
        int    num_gpus,
        float *weitage = NULL,
        bool   _enable_backward = false,
        bool   _keep_order      = false,
        bool   _keep_node_num   = false) :
        BasePartitioner(
            _enable_backward,
            _keep_order,
            _keep_node_num)
    {
        Init2(graph,num_gpus,weitage);
    }

    void Init2(
        const GraphT &graph,
        int num_gpus,
        float *weitage)
    {
        this->Init(graph,num_gpus);
        this->weitage=new float[num_gpus+1];
        if (weitage==NULL)
            for (int gpu=0;gpu<num_gpus;gpu++) this->weitage[gpu]=1.0f/num_gpus;
        else {
            float sum=0;
            for (int gpu=0;gpu<num_gpus;gpu++) sum+=weitage[gpu];
            for (int gpu=0;gpu<num_gpus;gpu++) this->weitage[gpu]=weitage[gpu]/sum;
        }
        for (int gpu=0;gpu<num_gpus;gpu++) this->weitage[gpu+1]+=this->weitage[gpu];
    }

    ~ParMetisPartitioner()
    {
        if (weitage!=NULL)
        {
            delete[] weitage;weitage=NULL;
        }
    }

    cudaError_t Partition(
        GraphT*    &sub_graphs,
        int**      &partition_tables,
        VertexId** &convertion_tables,
        VertexId** &original_vertexes,
        //SizeT**    &in_offsets,
        SizeT**    &in_counter,
        SizeT**    &out_offsets,
        SizeT**    &out_counter,
        SizeT**    &backward_offsets,
        int**      &backward_partitions,
        VertexId** &backward_convertions,
        float      factor = -1,
        int        seed   = -1)
    {
        cudaError_t retval = cudaSuccess;
        //typedef idxtype idx_t;
        idx_t       nodes   = this->graph->nodes;
        idx_t       edges   = this->graph->edges;
        idx_t       *vtxdist;
        idx_t       *vwgt   = NULL;
        idx_t       *adjwgt = NULL;
        idx_t       wgtflag = 0;
        idx_t       numflag = 0;
        idx_t       ngpus   = this->num_gpus;
        idx_t       ncons   = 1;
        float       *tpwgts;  // The fraction of vertex weights assigned to each partition
        float       *ubvec;   // The balance intolerance for vertex weights
        idx_t       options[3]; options[0] = 0;  // Enables options 2 and 3
                    // options[1] = 0;  // Timing information can be obtained by setting this to 1
                    // options[2] = 0;  // Random number seed for the routine
        idx_t       objval;
        idx_t*      tpartition_table = new idx_t[nodes];//=this->partition_tables[0];
        idx_t*      trow_offsets     = new idx_t[nodes+1];
        idx_t*      tcolumn_indices  = new idx_t[edges];


        for (idx_t node = 0; node <= nodes; node++)
            trow_offsets[node] = this->graph->row_offsets[node];
        for (idx_t edge = 0; edge < edges; edge++)
            tcolumn_indices[edge] = this->graph->column_indices[edge];

        tpwgts     = new float[ncons*ngpus];
        for(int p = 0; p < ngpus; ++p)
          tpwgts[p] = 1.0/ngpus;

        ubvec      = new float[ncons];
        ubvec[0]   = 1.05;

        MPI_Init(NULL, NULL);
        MPI_Comm comm;
        MPI_Comm_dup(MPI_COMM_WORLD, &comm);


        vtxdist    = new idx_t[ngpus+1];
        vtxdist[0] = 0;

        MPI_Allgather(&nodes, 1, MPI_INT, &vtxdist[1], 1, MPI_INT, comm);
        for(int p = 2; p <= ngpus; ++p)
         vtxdist[p] += vtxdist[p-1];

#ifdef PARMETIS_FOUND
      {
        //int Status =
                ParMETIS_V3_PartKway(
                    vtxdist,                     // vtxdist: distribution of vertices across processes
                    trow_offsets,                // xadj   : the adjacency structure of the graph
                    tcolumn_indices,             // adjncy : the adjacency structure of the graph
                    vwgt,                        // vwgt   : the weights of the vertices
                    adjwgt,                      // adjwgt : the weights of the edges
                    &wgtflag,                    // wgtflag: indicates which weights are present
                    &numflag,                    // numflag: indicates initial offset (0 or 1)
                    &ncons,                      // ncon   : the number of balancing constraints
                    &ngpus,                      // nparts : the number of parts to partition the graph
                    tpwgts,                      // tpwgts : the desired weight for each partition and constraint
                    ubvec,                       // ubvec  : the allowed load imbalance tolerance 4 each constraint
                    options,                     // options: the options setting random seed and timing information
                    &objval,                     // edgecut: the returned edgecut or the total communication volume
                    tpartition_table,            // part   : the returned partition vector of the graph
                    &comm);                      // comm   : the pointer to the MPI communicator (MPI_COMM_WORLD)

        // std::cout << "ParMetis: edgecut is " << objval << std::endl;

        for (SizeT i=0;i<nodes;i++) this->partition_tables[0][i]=tpartition_table[i];
        delete[] tpartition_table; tpartition_table = NULL;
        delete[] trow_offsets    ; trow_offsets     = NULL;
        delete[] tcolumn_indices ; tcolumn_indices  = NULL;

        retval = this->MakeSubGraph();
        sub_graphs           = this->sub_graphs;
        partition_tables     = this->partition_tables;
        convertion_tables    = this->convertion_tables;
        original_vertexes    = this->original_vertexes;
        //in_offsets           = this->in_offsets;
        in_counter           = this->in_counter;
        out_offsets          = this->out_offsets;
        out_counter          = this->out_counter;
        backward_offsets     = this->backward_offsets;
        backward_partitions  = this->backward_partitions;
        backward_convertions = this->backward_convertions;

      }
#else
      {
        const char * str = "ParMetis was not found during installation, therefore parmetis partitioner cannot be used.";
        retval = util::GRError(cudaErrorUnknown, str, __FILE__, __LINE__);
      } // METIS_FOUND
#endif
        return retval;
    }
};

} //namespace parmetis
} //namespace app
} //namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
