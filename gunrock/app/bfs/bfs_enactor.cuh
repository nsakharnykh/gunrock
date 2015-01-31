// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * bfs_enactor.cuh
 *
 * @brief BFS Problem Enactor
 */

#pragma once

#include <gunrock/util/multithreading.cuh>
#include <gunrock/util/multithread_utils.cuh>
#include <gunrock/util/kernel_runtime_stats.cuh>
#include <gunrock/util/test_utils.cuh>
//#include <gunrock/util/scan/multi_scan.cuh>

#include <gunrock/oprtr/advance/kernel.cuh>
#include <gunrock/oprtr/advance/kernel_policy.cuh>
#include <gunrock/oprtr/filter/kernel.cuh>
#include <gunrock/oprtr/filter/kernel_policy.cuh>

#include <gunrock/app/enactor_base.cuh>
#include <gunrock/app/bfs/bfs_problem.cuh>
#include <gunrock/app/bfs/bfs_functor.cuh>

#include <moderngpu.cuh>

namespace gunrock {
namespace app {
namespace bfs {

    template <typename BfsProblem, bool INSTRUMENT, bool DEBUG, bool SIZE_CHECK> class BFSEnactor;
        
    template <typename VertexId, typename SizeT, SizeT num_associates>
    __global__ void Expand_Incoming (
        const SizeT            num_elements,
        //const SizeT            incoming_offset,
        const VertexId*  const keys_in,
              VertexId*        keys_out,
              //unsigned int*    marker,
              VertexId**       associate_in,
              VertexId**       associate_org)
    {
        const SizeT STRIDE = gridDim.x * blockDim.x;
        __shared__ VertexId* s_associate_in [num_associates];
        __shared__ VertexId* s_associate_org[num_associates];
        //__shared__ VertexId* s_associate_in[2];
        //__shared__ VertexId* s_associate_org[2];
        //SizeT x2;
        if (threadIdx.x < num_associates)
        {
            s_associate_in [threadIdx.x]=associate_in [threadIdx.x];
            s_associate_org[threadIdx.x]=associate_org[threadIdx.x];
        }
        __syncthreads();

        VertexId key,t;
        SizeT x= blockIdx.x * blockDim.x + threadIdx.x;
        while (x<num_elements)
        {
            key = keys_in[x];
            t   = s_associate_in[0][x];

            //printf("\t %d,%d,%d,%d,%d ",x2,key,t,associate_org[0][key],marker[key]);
            if (atomicCAS(s_associate_org[0]+key, -1, t)== -1)
            {
            } else {
               if (atomicMin(s_associate_org[0]+key, t)<t)
               {
                   keys_out[x]=-1;
                   x+=STRIDE;
                   continue;
               }
            }
            //if (marker[key]==0) 
            //if (atomicCAS(marker+key, 0, 1)==0)
            //{
                //marker[key]=1;
                keys_out[x]=key;
            //} else keys_out[x]=-1;
            if (num_associates==2) 
                s_associate_org[1][key]=s_associate_in[1][x];
            /*#pragma unroll
            for (SizeT i=1;i<num_associates;i++)
            {
                associate_org[i][key]=associate_in[i][x2];
            }*/
            x+=STRIDE;
        }
    }

/*    template <typename BfsProblem>
    void ShowDebugInfo(
        int                    thread_num,
        int                    peer_,
        FrontierAttribute<typename BfsProblem::SizeT>      *frontier_attribute,
        EnactorStats           *enactor_stats,
        typename BfsProblem::DataSlice  *data_slice,
        typename BfsProblem::GraphSlice *graph_slice,
        util::CtaWorkProgressLifetime *work_progress,
        std::string            check_name = "",
        cudaStream_t           stream = 0)
    {
        typedef typename BfsProblem::SizeT    SizeT;
        typedef typename BfsProblem::VertexId VertexId;
        typedef typename BfsProblem::Value    Value;
        SizeT queue_length;

        //util::cpu_mt::PrintMessage(check_name.c_str(), thread_num, enactor_stats->iteration);
        //printf("%d \t %d\t \t reset = %d, index = %d\n",thread_num, enactor_stats->iteration, frontier_attribute->queue_reset, frontier_attribute->queue_index);fflush(stdout);
        //if (frontier_attribute->queue_reset) 
            queue_length = frontier_attribute->queue_length;
        //else if (enactor_stats->retval = util::GRError(work_progress->GetQueueLength(frontier_attribute->queue_index, queue_length, false, stream), "work_progress failed", __FILE__, __LINE__)) return;
	//util::cpu_mt::PrintCPUArray<SizeT, SizeT>((check_name+" Queue_Length").c_str(), &(queue_length), 1, thread_num, enactor_stats->iteration);
	printf("%d\t %lld\t %d\t stage%d\t %s\t Queue_Length = %d\n", thread_num, enactor_stats->iteration, peer_, data_slice->stages[peer_], check_name.c_str(), queue_length);fflush(stdout);
        //printf("%d \t %d\t \t peer_ = %d, selector = %d, length = %d, p = %p\n",thread_num, enactor_stats->iteration, peer_, frontier_attribute->selector,queue_length,graph_slice->frontier_queues[peer_].keys[frontier_attribute->selector].GetPointer(util::DEVICE));fflush(stdout);
        //util::cpu_mt::PrintGPUArray<SizeT, VertexId>((check_name+" keys").c_str(), graph_slice->frontier_queues[peer_].keys[frontier_attribute->selector].GetPointer(util::DEVICE), queue_length, thread_num, enactor_stats->iteration,-1, stream);
	//if (graph_slice->frontier_queues.values[frontier_attribute->selector].GetPointer(util::DEVICE)!=NULL)
	//    util::cpu_mt::PrintGPUArray<SizeT, Value   >("valu1", graph_slice->frontier_queues.values[frontier_attribute->selector].GetPointer(util::DEVICE), _queue_length, thread_num, enactor_stats->iteration);
	//util::cpu_mt::PrintGPUArray<SizeT, VertexId>("labe1", data_slice[0]->labels.GetPointer(util::DEVICE), graph_slice->nodes, thread_num, enactor_stats->iteration);
	//if (BFSProblem::MARK_PREDECESSORS)
	//    util::cpu_mt::PrintGPUArray<SizeT, VertexId>("pred1", data_slice[0]->preds.GetPointer(util::DEVICE), graph_slice->nodes, thread_num, enactor_stats->iteration);
	//if (BFSProblem::ENABLE_IDEMPOTENCE)
	//    util::cpu_mt::PrintGPUArray<SizeT, unsigned char>("mask1", data_slice[0]->visited_mask.GetPointer(util::DEVICE), (graph_slice->nodes+7)/8, thread_num, enactor_stats->iteration);
    }*/
     
    template<
        typename AdvanceKernelPolicy,
        typename FilterKernelPolicy,
        typename BfsEnactor>
    void BFSCore(
        int                              thread_num,
        int                              peer_,
        FrontierAttribute<typename BfsEnactor::SizeT>      
                                        *frontier_attribute,
        EnactorStats                    *enactor_stats,
        typename BfsEnactor::BfsProblem::DataSlice  *data_slice,
        typename BfsEnactor::BfsProblem::DataSlice  *d_data_slice,
        typename BfsEnactor::BfsProblem::GraphSlice *graph_slice,
        util::CtaWorkProgressLifetime   *work_progress,
        ContextPtr                       context,
        cudaStream_t                     stream)
    {
        typedef typename BfsEnactor::BfsProblem BfsProblem;
        typedef typename BfsEnactor::SizeT      SizeT;
        typedef typename BfsEnactor::VertexId   VertexId;
        typedef typename BfsEnactor::Value      Value;
        typedef typename BfsProblem::DataSlice  DataSlice;
        typedef typename BfsProblem::GraphSlice GraphSlice;
        //typedef BFSEnactor<BFSProblem, INSTRUMENT> BfsEnactor;
        typedef BFSFunctor<VertexId, SizeT, VertexId, BfsProblem> BfsFunctor;
        static const bool DEBUG      = BfsEnactor::DEBUG;
        static const bool INSTRUMENT = BfsEnactor::INSTRUMENT;

        if (frontier_attribute->queue_reset && frontier_attribute->queue_length ==0) 
        {
            work_progress->SetQueueLength(frontier_attribute->queue_index, 0, false, stream);
            if (DEBUG) util::cpu_mt::PrintMessage("return-1", thread_num, enactor_stats->iteration);
            return;
        }
        if (DEBUG) util::cpu_mt::PrintMessage("Advance begin",thread_num, enactor_stats->iteration);
        /*if (enactor_stats->retval = work_progress->SetQueueLength(frontier_attribute->queue_index+1,0,false,stream)) 
        {
            if (DEBUG) util::cpu_mt::PrintMessage("return0", thread_num, enactor_stats->iteration);
            return;
        }*/
        //int queue_selector = (data_slice->num_gpus>1 && peer_==0 && enactor_stats->iteration>0)?data_slice->num_gpus:peer_;
        //printf("%d\t %d \t \t peer_ = %d, selector = %d, length = %d, index = %d\n",thread_num, enactor_stats->iteration, peer_, queue_selector,frontier_attribute->queue_length, frontier_attribute->queue_index);fflush(stdout); 

        // Edge Map
        gunrock::oprtr::advance::LaunchKernel<AdvanceKernelPolicy, BfsProblem, BfsFunctor>(
            enactor_stats[0],
            frontier_attribute[0],
            d_data_slice,
            (VertexId*)NULL,
            (bool*)    NULL,
            (bool*)    NULL,
            data_slice ->scanned_edges  [peer_]                                       .GetPointer(util::DEVICE),
            graph_slice->frontier_queues[peer_].keys  [frontier_attribute->selector  ].GetPointer(util::DEVICE), 
            graph_slice->frontier_queues[peer_].keys  [frontier_attribute->selector^1].GetPointer(util::DEVICE),          
            (VertexId*)NULL,          // d_pred_in_queue
            graph_slice->frontier_queues[peer_].values[frontier_attribute->selector^1].GetPointer(util::DEVICE),          
            graph_slice->row_offsets   .GetPointer(util::DEVICE),
            graph_slice->column_indices.GetPointer(util::DEVICE),
            (SizeT*)   NULL,
            (VertexId*)NULL,
            graph_slice->nodes, //frontier_queues[queue_selector].keys[frontier_attribute->selector  ].GetSize(), /
            graph_slice->edges, //frontier_queues[peer_         ].keys[frontier_attribute->selector^1].GetSize(), 
            work_progress[0],
            context[0],
            stream,
            gunrock::oprtr::advance::V2V,
            false,
            false);
        
        // Only need to reset queue for once
        //if (frontier_attribute->queue_reset)
        frontier_attribute->queue_reset = false;
        //if (DEBUG && (enactor_stats->retval = util::GRError(cudaStreamSynchronize(stream), "cudaStreamSynchronize failed", __FILE__, __LINE__))) 
        //{
        //    util::cpu_mt::PrintMessage("return1", thread_num, enactor_stats->iteration);
        //    return;
        //}
        //cudaStreamSynchronize(stream);
        //if (DEBUG && (enactor_stats->retval = util::GRError("advance::Kernel failed", __FILE__, __LINE__))) 
        //{
        //    util::cpu_mt::PrintMessage("return2", thread_num, enactor_stats->iteration);
        //    return;
        //}
        if (DEBUG) util::cpu_mt::PrintMessage("Advance end", thread_num, enactor_stats->iteration);
        frontier_attribute->queue_index++;
        frontier_attribute->selector ^= 1;
        if (false) //(DEBUG || INSTRUMENT)
        {
            if (enactor_stats->retval = work_progress->GetQueueLength(frontier_attribute->queue_index, frontier_attribute->queue_length,false,stream)) return;
            enactor_stats->total_queued += frontier_attribute->queue_length;
            if (DEBUG) ShowDebugInfo<BfsProblem>(thread_num, peer_, frontier_attribute, enactor_stats, data_slice, graph_slice, work_progress, "post_advance", stream);
            if (INSTRUMENT) {
                if (enactor_stats->retval = enactor_stats->advance_kernel_stats.Accumulate(
                    enactor_stats->advance_grid_size,
                    enactor_stats->total_runtimes,
                    enactor_stats->total_lifetimes,
                    false,stream)) return;
            }
        }

        //if (enactor_stats->retval = work_progress->SetQueueLength(frontier_attribute->queue_index+1, 0, false, stream)) return; 
        if (DEBUG) util::cpu_mt::PrintMessage("Filter begin", thread_num, enactor_stats->iteration);

        // Filter
        gunrock::oprtr::filter::Kernel<FilterKernelPolicy, BfsProblem, BfsFunctor>
        <<<enactor_stats->filter_grid_size, FilterKernelPolicy::THREADS, 0, stream>>>(
            enactor_stats->iteration+1,
            frontier_attribute->queue_reset,
            frontier_attribute->queue_index,
            //num_gpus,
            frontier_attribute->queue_length,
            //enactor_stats->d_done,
            graph_slice->frontier_queues[peer_].keys  [frontier_attribute->selector  ].GetPointer(util::DEVICE),      // d_in_queue
            graph_slice->frontier_queues[peer_].values[frontier_attribute->selector  ].GetPointer(util::DEVICE),    // d_pred_in_queue
            graph_slice->frontier_queues[peer_].keys  [frontier_attribute->selector^1].GetPointer(util::DEVICE),    // d_out_queue
            d_data_slice,
            data_slice->visited_mask.GetPointer(util::DEVICE),
            work_progress[0],
            graph_slice->frontier_queues[peer_].keys  [frontier_attribute->selector  ].GetSize(),
            //graph_slice->frontier_elements[peer_][frontier_attribute->selector  ],           // max_in_queue
            graph_slice->frontier_queues[peer_].keys  [frontier_attribute->selector^1].GetSize(),
            //graph_slice->frontier_elements[peer_][frontier_attribute->selector^1],         // max_out_queue
            enactor_stats->filter_kernel_stats);
	    //t_bitmask);

        //cudaStreamSynchronize(stream);
        //if (DEBUG && (enactor_stats->retval = util::GRError(cudaStreamSynchronize(stream), "cudaStreamSynchronize failed", __FILE__, __LINE__))) return;
	if (DEBUG && (enactor_stats->retval = util::GRError("filter_forward::Kernel failed", __FILE__, __LINE__))) return;
	if (DEBUG) util::cpu_mt::PrintMessage("Filter end.", thread_num, enactor_stats->iteration);
	frontier_attribute->queue_index++;
	frontier_attribute->selector ^= 1;
	if (false) {//(INSTRUMENT || DEBUG) {
	    //if (enactor_stats->retval = work_progress->GetQueueLength(frontier_attribute->queue_index, frontier_attribute->queue_length)) break;
	    //enactor_stats->total_queued += frontier_attribute->queue_length;
            if (DEBUG) ShowDebugInfo<BfsProblem>(thread_num, peer_, frontier_attribute, enactor_stats, data_slice, graph_slice, work_progress, "post_filter", stream);
	    if (INSTRUMENT) {
		if (enactor_stats->retval = enactor_stats->filter_kernel_stats.Accumulate(
		    enactor_stats->filter_grid_size,
		    enactor_stats->total_runtimes,
		    enactor_stats->total_lifetimes,
		    false, stream)) return;
	    }
	}
    }

    template<
        typename AdvanceKernelPolicy,
        typename FilterKernelPolicy,
        //typename BFSProblem,
        typename BfsEnactor>
    static CUT_THREADPROC BFSThread(
        void * thread_data_)
    {
        typedef typename BfsEnactor::BfsProblem BfsProblem;
        typedef typename BfsEnactor::SizeT      SizeT     ;
        typedef typename BfsEnactor::VertexId   VertexId  ;
        typedef typename BfsEnactor::Value      Value     ;
        typedef typename BfsProblem::DataSlice  DataSlice ;
        typedef typename BfsProblem::GraphSlice GraphSlice;
        //typedef BFSEnactor<BFSProblem, INSTRUMENT, DEBUG, SIZE_CHECK> BfsEnactor;
        typedef BFSFunctor<VertexId, SizeT, VertexId, BfsProblem> BfsFunctor;
        //static const bool INSTRUMENT = BfsEnactor::INSTRUMENT;
        static const bool DEBUG      = BfsEnactor::DEBUG     ;
        static const bool SIZE_CHECK = BfsEnactor::SIZE_CHECK;

        ThreadSlice  *thread_data          = (ThreadSlice *) thread_data_;
        BfsProblem   *problem              = (BfsProblem*) thread_data->problem;
        BfsEnactor   *enactor              = (BfsEnactor*) thread_data->enactor;
        int          num_gpus              =   problem     -> num_gpus;
        int          thread_num            =   thread_data -> thread_num;
        int          gpu_idx               =   problem     -> gpu_idx           [thread_num];
        DataSlice    *data_slice           =   problem     -> data_slices       [thread_num].GetPointer(util::HOST);
        util::Array1D<SizeT, DataSlice>
                     *s_data_slice         =   problem     -> data_slices;
        GraphSlice   *graph_slice          =   problem     -> graph_slices      [thread_num];
        GraphSlice   **s_graph_slice       =   problem     -> graph_slices;
        FrontierAttribute<SizeT>
                     *frontier_attribute   = &(enactor     -> frontier_attribute[thread_num*num_gpus]);
        FrontierAttribute<SizeT>
                     *s_frontier_attribute = &(enactor     -> frontier_attribute[0         ]);
        EnactorStats *enactor_stats        = &(enactor     -> enactor_stats     [thread_num*num_gpus]);
        EnactorStats *s_enactor_stats      = &(enactor     -> enactor_stats     [0         ]);
        util::CtaWorkProgressLifetime
                     *work_progress        = &(enactor     -> work_progress     [thread_num*num_gpus]);
        ContextPtr*  context               =   thread_data -> context;
        int*         stages                =   data_slice  -> stages .GetPointer(util::HOST);
        bool*        to_show               =   data_slice  -> to_show.GetPointer(util::HOST);
        cudaStream_t* streams              =   data_slice  -> streams.GetPointer(util::HOST);
        SizeT        Total_Length          = 0;
        //bool         First_Stage4          = true; 
        cudaError_t  tretval               = cudaSuccess;
        int          grid_size             = 0;
        std::string  mssg                  = "";
        int          pre_stage             = 0;
        size_t       offset                = 0;
        int          num_vertex_associate  = BfsProblem::MARK_PREDECESSORS? 2:1;
        int          num_value__associate  = 0;  
 
        do {
            //util::cpu_mt::PrintMessage("BFS Thread begin.",thread_num, enactor_stats[0].iteration);
            if (enactor_stats[0].retval = util::SetDevice(gpu_idx)) break;
            thread_data->stats = 1;
            while (thread_data->stats != 2) sleep(0);
            thread_data->stats=3;

            for (int peer=0;peer<num_gpus;peer++)
            {
                frontier_attribute[peer].queue_index    = 0;        // Work queue index
                frontier_attribute[peer].selector       = 0;
                frontier_attribute[peer].queue_length   = peer==0?thread_data -> init_size:0; 
                frontier_attribute[peer].queue_reset    = true;
                enactor_stats[peer].iteration           = 0;
            }

            // Step through BFS iterations
            while (!All_Done<SizeT, DataSlice>(s_enactor_stats, s_frontier_attribute, s_data_slice, num_gpus)) 
            {                
                if (num_gpus>1 && enactor_stats[0].iteration>0)
                {
                    //if (DEBUG) ShowDebugInfo<BFSProblem>(thread_num, 0, frontier_attribute, enactor_stats, data_slice, graph_slice, work_progress, std::string("post_scan"));
                    //if (DEBUG) util::cpu_mt::PrintGPUArray<SizeT, VertexId>("keys0",graph_slice->frontier_queues[num_gpus].keys[frontier_attribute[0].selector].GetPointer(util::DEVICE), frontier_attribute[0].queue_length, thread_num, enactor_stats[0].iteration);
                    frontier_attribute[0].queue_reset  = true;
                    frontier_attribute[0].queue_offset = 0;
                    //frontier_attribute[0].queue_length = Total_Length;
                    for (int i=1;i<num_gpus;i++)
                    {
                        frontier_attribute[i].selector      = frontier_attribute[0].selector;
                        frontier_attribute[i].advance_type  = frontier_attribute[0].advance_type;
                        //frontier_attribute[i].queue_length  = Total_Length;
                        //frontier_attribute[i].queue_length  = data_slice->out_length[i];
                        //frontier_attribute[i].queue_offset  = frontier_attribute[i-1].queue_offset + data_slice->out_length[i-1];
                        frontier_attribute[i].queue_offset  = 0;
                        frontier_attribute[i].queue_reset   = true;
                        frontier_attribute[i].queue_index   = frontier_attribute[0].queue_index;
                        frontier_attribute[i].current_label = frontier_attribute[0].current_label;
                        enactor_stats     [i].iteration     = enactor_stats     [0].iteration;
                    }
                    //frontier_attribute[0].queue_length = data_slice->out_length[0];
                } else {
                    frontier_attribute[0].queue_offset = 0;
                    frontier_attribute[0].queue_reset  = true;
                }
              
                Total_Length      = 0;
                //First_Stage4      = true; 
                data_slice->wait_counter= 0;
                tretval           = cudaSuccess;
                //to_wait           = false;
                for (int peer=0;peer<num_gpus;peer++)
                {
                    stages [peer] = 0   ; stages [peer+num_gpus]=0;
                    to_show[peer] = true; to_show[peer+num_gpus]=true;
                    for (int i=0;i<data_slice->num_stages;i++)
                        data_slice->events_set[enactor_stats[0].iteration%4][peer][i]=false;
                }

                while (data_slice->wait_counter <num_gpus*2 
                       && (!All_Done(s_enactor_stats, s_frontier_attribute, s_data_slice, num_gpus)))
                {
                    for (int peer__=0;peer__<num_gpus*2;peer__++)
                    {
                        //if (peer__==num_gpus) continue;
                        int peer_ = (peer__%num_gpus);
                        int peer = peer_<= thread_num? peer_-1   : peer_       ;
                        int gpu_ = peer <  thread_num? thread_num: thread_num+1;
                        if (DEBUG && to_show[peer__])
                        {
                            //util::cpu_mt::PrintCPUArray<SizeT, int>("stages",data_slice->stages.GetPointer(util::HOST),num_gpus,thread_num,enactor_stats[peer_].iteration);
                            //mssg="pre_stage0";
                            //mssg[9]=char(stages[peer_]+'0');
                            mssg="";
                            ShowDebugInfo<BfsProblem>(
                                thread_num, 
                                peer__, 
                                &frontier_attribute[peer_], 
                                &enactor_stats[peer_], 
                                data_slice, 
                                graph_slice, 
                                &work_progress[peer_], 
                                mssg,
                                streams[peer__]);
                        }
                        to_show[peer__]=true;
                        int iteration  = enactor_stats[peer_].iteration;
                        int iteration_ = iteration%4;
                        pre_stage      = stages[peer__];
                        //int queue_selector = 0;
                        int selector   = frontier_attribute[peer_].selector;
                        util::DoubleBuffer<SizeT, VertexId, VertexId> 
                            *frontier_queue_ = &(graph_slice->frontier_queues[peer_]);
                        FrontierAttribute<SizeT>* frontier_attribute_ = &(frontier_attribute[peer_]); 
                        EnactorStats* enactor_stats_ = &(enactor_stats[peer_]);

                        switch (stages[peer__])
                        {
                        case 0: // Assign marker & Scan
                            if (peer_==0) {
                                //cudaEventRecord(data_slice->events[iteration_][peer_][6],streams[peer_]);
                                //data_slice->events_set[iteration_][peer_][6]=true; 
                                //printf("%d\t %d\t %d\t jump to stage 7\n", thread_num, iteration, peer_);fflush(stdout);
                                if (peer__==num_gpus || frontier_attribute_->queue_length==0) stages[peer__]=3;
                                break;
                            } else if ((iteration==0 || data_slice->out_length[peer_]==0) && peer__>num_gpus) {
                                cudaEventRecord(data_slice->events[iteration_][peer_][0],streams[peer__]);
                                //cudaEventRecord(data_slice->events[iteration_][peer_][6],streams[peer_]);
                                data_slice->events_set[iteration_][peer_][0]=true;
                                //data_slice->events_set[iteration_][peer_][6]=true;
                                //frontier_attribute_->queue_length=0;
                                //printf("%d\t %d\t %d\t jump to stage 7\n", thread_num, iteration, peer_);fflush(stdout);
                                stages[peer__]=3;break;
                            //} else if (num_gpus<2 || (peer_==0 && iteration==0)) {
                            //    //printf("%d\t %d\t %d\t jump to stage 4\n", thread_num, iteration, peer_);fflush(stdout);
                            //    break;
                            } /*else if (peer_>0 && frontier_attribute_->queue_length==0) {
                                cudaEventRecord(data_slice->events[iteration_][peer_][2],streams[peer_]);
                                data_slice->events_set[iteration_][peer_][2]=true;
                                data_slice->out_length[peer_]=0;
                                //printf("%d\t %d\t %d\t jump to stage 3\n", thread_num, iteration, peer_);fflush(stdout);
                                stages[peer_]=2;break;
                            }*/
                            //printf("%d\t %d\t %d\t entered stage0\n", thread_num, iteration, peer_);fflush(stdout);

                            if (peer__<num_gpus) 
                            { //wait and expand incoming
                                if (!(s_data_slice[peer]->events_set[iteration_][gpu_][0]))
                                {   to_show[peer__]=false;stages[peer__]--;break;}
                           
                                frontier_attribute_->queue_length = data_slice->in_length[iteration%2][peer_];
                                data_slice->in_length[iteration%2][peer_]=0;
                                if (frontier_attribute_->queue_length ==0)
                                {
                                    //cudaEventRecord(data_slice->events[iteration_][peer_][6],streams[peer_]);
                                    //data_slice->events_set[iteration_][peer_][6]=true; 
                                    //printf("%d\t %d\t %d\t jump to stage 7\n", thread_num, iteration, peer_);fflush(stdout);
                                    stages[peer__]=3;break;                                
                                }

                                if (frontier_attribute_->queue_length > frontier_queue_->keys[selector^1].GetSize())
                                {
                                    printf("%d\t %d\t %d\t queue1   \t oversize :\t %d ->\t %d\n", 
                                        thread_num, iteration, peer_, 
                                        frontier_queue_->keys[selector^1].GetSize(),
                                        frontier_attribute_->queue_length);
                                        fflush(stdout);
                                    if (SIZE_CHECK)
                                    {
                                        if (enactor_stats_->retval = frontier_queue_->keys[selector^1].EnsureSize(frontier_attribute_->queue_length)) break;
                                        if (BfsProblem::USE_DOUBLE_BUFFER)
                                        {
                                            if (enactor_stats_->retval = frontier_queue_->values[selector^1].EnsureSize(frontier_attribute_->queue_length)) break;
                                        }
                                    } else {
                                        enactor_stats_->retval = util::GRError(cudaErrorLaunchOutOfResources, "queue1 oversize", __FILE__, __LINE__);
                                        break;
                                    }
                                }

                                grid_size = frontier_attribute_->queue_length/256+1;
                                if (grid_size>512) grid_size=512;
                                //cudaStreamSynchronize(data_slice->streams[peer_]);
                                //if (enactor_stats[peer_].retval = util::GRError("cudaStreamSynchronize failed", __FILE__, __LINE__)) break;
                                //if (enactor_stats[0].iteration==2) break;
                                cudaStreamWaitEvent(streams[peer_], 
                                    s_data_slice[peer]->events[iteration_][gpu_][0], 0);
                                Expand_Incoming <VertexId, SizeT, BfsProblem::MARK_PREDECESSORS?2:1>
                                    <<<grid_size,256,0,streams[peer_]>>> (
                                    frontier_attribute_->queue_length,
                                    //graph_slice ->in_offset[peer_],
                                    data_slice ->keys_in             [iteration%2][peer_].GetPointer(util::DEVICE),
                                    frontier_queue_->keys                    [selector^1].GetPointer(util::DEVICE),
                                    //data_slice ->temp_marker                             .GetPointer(util::DEVICE),
                                    data_slice ->vertex_associate_ins[iteration%2][peer_].GetPointer(util::DEVICE),
                                    data_slice ->vertex_associate_orgs                   .GetPointer(util::DEVICE));
                                frontier_attribute_->selector^=1;
                                frontier_attribute_->queue_index++;
 
                            } else { //Push Neibor
                                PushNeibor <SIZE_CHECK, SizeT, VertexId, Value, GraphSlice, DataSlice, 
                                        BfsProblem::MARK_PREDECESSORS?2:1, 0> (
                                    thread_num,
                                    peer, 
                                    data_slice->out_length[peer_],
                                    enactor_stats_,
                                    s_data_slice  [thread_num].GetPointer(util::HOST),
                                    s_data_slice  [peer]      .GetPointer(util::HOST),
                                    s_graph_slice [thread_num],
                                    s_graph_slice [peer],
                                    streams       [peer__]);
                                cudaEventRecord(data_slice->events[iteration_][peer_][stages[peer__]],streams[peer__]);
                                data_slice->events_set[iteration_][peer_][stages[peer__]]=true;
                                stages[peer__]=3;
                            }
                            break;
 
                        case 1: //Comp Length                           
                            gunrock::oprtr::advance::ComputeOutputLength 
                                <AdvanceKernelPolicy, BfsProblem, BfsFunctor>(
                                frontier_attribute_,
                                graph_slice ->row_offsets     .GetPointer(util::DEVICE),
                                graph_slice ->column_indices  .GetPointer(util::DEVICE),
                                graph_slice ->frontier_queues  [peer_].keys[selector].GetPointer(util::DEVICE),
                                data_slice  ->scanned_edges    [peer_].GetPointer(util::DEVICE),
                                graph_slice ->nodes,//frontier_queues[peer_].keys[frontier_attribute[peer_].selector  ].GetSize(), 
                                graph_slice ->edges,//frontier_queues[peer_].keys[frontier_attribute[peer_].selector^1].GetSize(),
                                context          [peer_][0],
                                streams          [peer_],
                                gunrock::oprtr::advance::V2V, true);

                            if (SIZE_CHECK) 
                            {
                                frontier_attribute_->output_length.Move(util::DEVICE, util::HOST,1,0,streams[peer_]);
                                cudaEventRecord(data_slice->events[iteration_][peer_][stages[peer_]], streams[peer_]);
                                data_slice->events_set[iteration_][peer_][stages[peer_]]=true;
                            }
                            break;

                        case 2: //BFS Core
                            if (SIZE_CHECK)
                            {
                                if (!data_slice->events_set[iteration_][peer_][stages[peer_]-1])
                                {   to_show[peer_]=false;stages[peer_]--;break;}
                                tretval = cudaEventQuery(data_slice->events[iteration_][peer_][stages[peer_]-1]);
                                if (tretval == cudaErrorNotReady) 
                                {   to_show[peer_]=false;stages[peer_]--; break;} 
                                else if (tretval !=cudaSuccess) {enactor_stats_->retval=tretval; break;}

                                if (DEBUG) {printf("%d\t %d\t %d\t queue_length = %d, output_length = %d\n",
                                        thread_num, iteration, peer_,
                                        frontier_queue_->keys[selector^1].GetSize(), 
                                        frontier_attribute_->output_length[0]);fflush(stdout);}
                                //frontier_attribute_->output_length[0]+=1;
                                if (frontier_attribute_->output_length[0]+1 > frontier_queue_->keys[selector^1].GetSize())  
                                {
                                    printf("%d\t %d\t %d\t queue3   \t oversize :\t %d ->\t %d\n",
                                        thread_num, iteration, peer_,
                                        frontier_queue_->keys[selector^1].GetSize(), 
                                        frontier_attribute_->output_length[0]+1);fflush(stdout);
                                    if (enactor_stats_->retval = frontier_queue_->keys[selector  ].EnsureSize(frontier_attribute_->output_length[0]+1, true)) break;
                                    if (enactor_stats_->retval = frontier_queue_->keys[selector^1].EnsureSize(frontier_attribute_->output_length[0]+1)) break;
                                 
                                    if (BfsProblem::USE_DOUBLE_BUFFER) {
                                        if (enactor_stats_->retval = frontier_queue_->values[selector  ].EnsureSize(frontier_attribute_->output_length[0]+1,true)) break;
                                        if (enactor_stats_->retval = frontier_queue_->values[selector^1].EnsureSize(frontier_attribute_->output_length[0]+1)) break;
                                   cudaStreamSynchronize(0); 
                                   }
                                   //if (enactor_stats[peer_].retval = cudaDeviceSynchronize()) break;
                                }
                            }
      
                            BFSCore <AdvanceKernelPolicy, FilterKernelPolicy, BfsEnactor>(
                                thread_num,
                                peer_,
                                frontier_attribute_,
                                enactor_stats_,
                                data_slice,
                                s_data_slice[thread_num].GetPointer(util::DEVICE),
                                graph_slice,
                                &(work_progress[peer_]),
                                context[peer_],
                                streams[peer_]);
                            if (enactor_stats_->retval = work_progress[peer_].GetQueueLength(
                                frontier_attribute_->queue_index, 
                                frontier_attribute_->queue_length, 
                                false, 
                                streams[peer_], 
                                true)) break; 
                            if (num_gpus>1)
                            {
                                cudaEventRecord(data_slice->events[iteration_][peer_][stages[peer_]], streams[peer_]);
                                data_slice->events_set[iteration_][peer_][stages[peer_]]=true;
                            }
                            break;
                        
                        case 3: //Copy
                            if (num_gpus <=1) {to_show[peer_]=false;break;}
                            /*to_wait = false;
                            for (int i=0;i<num_gpus;i++)
                                if (stages[i]<stages[peer_])
                                {
                                    to_wait=true;break;
                                }
                            if (to_wait)*/
                            {
                                if (!data_slice->events_set[iteration_][peer_][stages[peer_]-1])
                                {   to_show[peer_]=false;stages[peer_]--;break;}
                                tretval = cudaEventQuery(data_slice->events[iteration_][peer_][stages[peer_]-1]);
                                if (tretval == cudaErrorNotReady) 
                                {   to_show[peer_]=false;stages[peer_]--;break;} 
                                else if (tretval !=cudaSuccess) {enactor_stats_->retval=tretval; break;}
                            } //else cudaStreamSynchronize(streams[peer_]);
                            //data_slice->events_set[iteration_][peer_][stages[peer_]-1]=false;

                            /*if (DEBUG) 
                            {
                                printf("%d\t %lld\t %d\t org_length = %d, queue_length = %d, new_length = %d, max_length = %d\n", 
                                    thread_num, 
                                    enactor_stats[peer_].iteration, 
                                    peer_, 
                                    Total_Length, 
                                    frontier_attribute[peer_].queue_length,
                                    Total_Length + frontier_attribute[peer_].queue_length,
                                    graph_slice->frontier_queues[num_gpus].keys[frontier_attribute[0].selector].GetSize());
                                fflush(stdout);
                            }*/
 
                            /*if (!SIZE_CHECK)     
                            {
                                //printf("output_length = %d, queue_size = %d\n", frontier_attribute[peer_].output_length[0], graph_slice->frontier_queues[peer_].keys[frontier_attribute[peer_].selector^1].GetSize());fflush(stdout);
                                if (frontier_attribute[peer_].output_length[0] > graph_slice->frontier_queues[peer_].keys[frontier_attribute[peer_].selector^1].GetSize())  
                                {
                                    printf("%d\t %lld\t %d\t queue3 oversize :\t %d ->\t %d\n",
                                        thread_num, enactor_stats[peer_].iteration, peer_,
                                        graph_slice->frontier_queues[peer_].keys[frontier_attribute[peer_].selector^1].GetSize(), 
                                        frontier_attribute[peer_].output_length[0]);fflush(stdout);
                                }
                            }*/
                            if (frontier_attribute_->queue_length!=0)
                            {
                                /*if (frontier_attribute[peer_].queue_length > 
                                    graph_slice->frontier_queues[peer_].keys[frontier_attribute[peer_].selector].GetSize())
                                {
                                   printf("%d\t %lld\t %d\t sub_queue oversize : queue_length = %d, queue_size = %d\n",
                                       thread_num, enactor_stats[peer_].iteration, peer_,
                                       frontier_attribute[peer_].queue_length, graph_slice->frontier_queues[peer_].keys[frontier_attribute[peer_].selector].GetSize());
                                   fflush(stdout);
                                }*/
                                if (Total_Length + frontier_attribute_->queue_length > graph_slice->frontier_queues[num_gpus].keys[0].GetSize())
                                {
                                    printf("%d\t %d\t %d\t total_queue\t oversize :\t %d ->\t %d \n",
                                       thread_num, iteration, peer_,
                                       Total_Length + frontier_attribute_->queue_length,
                                       graph_slice->frontier_queues[num_gpus].keys[0].GetSize());fflush(stdout);
                                    if (SIZE_CHECK)
                                    {
                                        if (enactor_stats_->retval = graph_slice->frontier_queues[num_gpus].keys[0].EnsureSize(Total_Length+frontier_attribute_->queue_length, true)) break;
                                        if (BfsProblem::USE_DOUBLE_BUFFER)
                                        {
                                            if (enactor_stats_->retval = graph_slice->frontier_queues[num_gpus].values[0].EnsureSize(Total_Length + frontier_attribute_->queue_length, true)) break;
                                        }
                                    } else {
                                        enactor_stats_->retval = util::GRError(cudaErrorLaunchOutOfResources, "total_queue oversize", __FILE__, __LINE__);
                                        break;
                                    }
                                }
                                util::MemsetCopyVectorKernel<<<256,256, 0, streams[peer_]>>>(
                                    graph_slice->frontier_queues[num_gpus].keys[0].GetPointer(util::DEVICE) + Total_Length, 
                                    frontier_queue_->keys[selector].GetPointer(util::DEVICE), 
                                    frontier_attribute_->queue_length);
                                if (BfsProblem::USE_DOUBLE_BUFFER)
                                    util::MemsetCopyVectorKernel<<<256,256,0,streams[peer_]>>>(
                                        graph_slice->frontier_queues[num_gpus].values[0].GetPointer(util::DEVICE) + Total_Length,
                                        frontier_queue_->values[selector].GetPointer(util::DEVICE),
                                        frontier_attribute_->queue_length);
                                Total_Length+=frontier_attribute_->queue_length;
                            }
                            /*if (First_Stage4)
                            {
                                First_Stage4=false;
                                util::MemsetKernel<<<256, 256, 0, streams[peer_]>>>
                                    (data_slice->temp_marker.GetPointer(util::DEVICE), 
                                    (unsigned int)0, graph_slice->nodes);
                            }*/
                            //cudaEventRecord(data_slice->events[iteration_][peer_][stages[peer_]], streams[peer_]);
                            //data_slice->events_set[iteration_][peer_][stages[peer_]]=true;
                            
                            break;

                        case 4: //End
                            data_slice->wait_counter++;
                            to_show[peer__]=false;
                            break;
                        default:
                            stages[peer__]--;
                            to_show[peer__]=false;
                        }
                        
                        if (DEBUG && !enactor_stats_->retval)
                        {
                            mssg="stage 0 @ gpu 0, peer_ 0 failed";
                            mssg[6]=char(pre_stage+'0');
                            mssg[14]=char(thread_num+'0');
                            mssg[23]=char(peer__+'0');
                            if (enactor_stats_->retval = util::GRError(//cudaStreamSynchronize(streams[peer_]),
                                 mssg, __FILE__, __LINE__)) break;
                            //sleep(1);
                        }
                        stages[peer__]++;
                        if (enactor_stats_->retval) break;
                        //if (All_Done(s_enactor_stats, s_frontier_attribute, s_data_slice, num_gpus)) break;
                    }
                    /*to_wait=true;
                    for (int i=0;i<num_gpus;i++)
                        if (to_show[i])
                        {
                            to_wait=false;
                            break;
                        }
                    if (to_wait) sleep(0);*/
                }
              
                if (num_gpus>1 && !All_Done<SizeT, DataSlice>(s_enactor_stats, s_frontier_attribute, s_data_slice, num_gpus))
                {
                    /*if (First_Stage4)
                    {
                        util::MemsetKernel<<<256,256, 0, streams[0]>>>
                            (data_slice->temp_marker.GetPointer(util::DEVICE),
                            (unsigned int)0, graph_slice->nodes);
                    }*/
                    for (int peer_=0;peer_<num_gpus;peer_++)
                    for (int i=0;i<data_slice->num_stages;i++)
                        data_slice->events_set[(enactor_stats[0].iteration+3)%4][peer_][i]=false;

                    for (int peer_=0;peer_<num_gpus*2;peer_++)
                        data_slice->wait_marker[peer_]=0;
                    int wait_count=0;
                    while (wait_count<num_gpus*2-1 && 
                        !All_Done<SizeT, DataSlice>(s_enactor_stats, s_frontier_attribute, s_data_slice, num_gpus))
                    {
                        for (int peer_=0;peer_<num_gpus*2;peer_++)
                        {
                            if (peer_==num_gpus || data_slice->wait_marker[peer_]!=0)
                                continue;
                            cudaError_t tretval = cudaStreamQuery(streams[peer_]);
                            if (tretval == cudaSuccess)
                            {
                                data_slice->wait_marker[peer_]=1;
                                wait_count++;
                                continue;
                            } else if (tretval != cudaErrorNotReady)
                            {
                                enactor_stats[peer_%num_gpus].retval = tretval;
                                break;
                            }
                        }
                    }
                    //printf("%d\t %lld\t past StreamSynchronize\n", thread_num, enactor_stats[0].iteration);
                    /*if (SIZE_CHECK)
                    {
                        if (graph_slice->frontier_queues[num_gpus].keys[frontier_attribute[0].selector^1].GetSize() < Total_Length)
                        {
                            printf("%d\t %lld\t \t keysn oversize : %d -> %d \n",
                               thread_num, enactor_stats[0].iteration,
                               graph_slice->frontier_queues[num_gpus].keys[frontier_attribute[0].selector^1].GetSize(), Total_Length);
                            if (enactor_stats[0].retval = graph_slice->frontier_queues[num_gpus].keys[frontier_attribute[0].selector^1].EnsureSize(Total_Length)) break;
                            if (BFSProblem::USE_DOUBLE_BUFFER)
                            {
                                if (enactor_stats[0].retval = graph_slice->frontier_queues[num_gpus].values[frontier_attribute[0].selector^1].EnsureSize(Total_Length)) break;
                            }
                        }
                    }*/
                    
                    frontier_attribute[0].queue_length = Total_Length;
                                       
                    if (Total_Length >0)
                    {
                        grid_size = Total_Length/256+1;
                        if (grid_size > 512) grid_size = 512;

                        if (BfsProblem::MARK_PREDECESSORS)
                        {
                            Copy_Preds<VertexId, SizeT> <<<grid_size,256,0, streams[0]>>>(
                                Total_Length,
                                graph_slice-> frontier_queues[num_gpus].keys[0].GetPointer(util::DEVICE),
                                data_slice -> preds.GetPointer(util::DEVICE),
                                data_slice -> temp_preds.GetPointer(util::DEVICE));
 
                            Update_Preds<VertexId,SizeT> <<<grid_size,256,0,streams[0]>>>(
                                Total_Length,
                                graph_slice->nodes,
                                graph_slice -> frontier_queues[num_gpus].keys[0].GetPointer(util::DEVICE),
                                graph_slice -> original_vertex.GetPointer(util::DEVICE),
                                data_slice  -> temp_preds     .GetPointer(util::DEVICE),
                                data_slice  -> preds          .GetPointer(util::DEVICE));
                        }

                        if (data_slice->keys_marker[0].GetSize() < Total_Length)
                        {
                            printf("%d\t %lld\t \t keys_marker\t oversize :\t %d ->\t %d \n",
                                    thread_num, enactor_stats[0].iteration,
                                    data_slice->keys_marker[0].GetSize(), Total_Length);fflush(stdout);       
                            if (SIZE_CHECK)
                            {
                                for (int peer_=0;peer_<num_gpus;peer_++)
                                {
                                    data_slice->keys_marker[peer_].EnsureSize(Total_Length);
                                    data_slice->keys_markers[peer_]=data_slice->keys_marker[peer_].GetPointer(util::DEVICE);
                                }
                                data_slice->keys_markers.Move(util::HOST, util::DEVICE, num_gpus, 0, streams[0]);
                            } else {
                                enactor_stats[0].retval = util::GRError(cudaErrorLaunchOutOfResources, "keys_marker oversize", __FILE__, __LINE__);
                                break;
                            }
                        }

                        Assign_Marker<VertexId, SizeT>
                            <<<grid_size,256, num_gpus * sizeof(SizeT*) ,streams[0]>>> (
                            Total_Length,
                            num_gpus,
                            graph_slice->frontier_queues[num_gpus].keys[0].GetPointer(util::DEVICE),
                            graph_slice->partition_table.GetPointer(util::DEVICE),
                            data_slice->keys_markers.GetPointer(util::DEVICE));

                        for (int peer_=0;peer_<num_gpus;peer_++)
                        {
                            Scan<mgpu::MgpuScanTypeInc>(
                                (int*)data_slice->keys_marker[peer_].GetPointer(util::DEVICE),
                                Total_Length, 
                                (int)0, mgpu::plus<int>(), (int*)0, (int*)0, 
                                (int*)data_slice->keys_marker[peer_].GetPointer(util::DEVICE), 
                                context[0][0]);
                        }
                            
                        for (int peer_=0; peer_<num_gpus;peer_++)
                        {
                            cudaMemcpyAsync(&(data_slice->out_length[peer_]), 
                                data_slice->keys_marker[peer_].GetPointer(util::DEVICE) 
                                     + (Total_Length -1), 
                                sizeof(SizeT), cudaMemcpyDeviceToHost, streams[0]);
                        }
                        cudaStreamSynchronize(streams[0]);
                            
                        for (int peer_=0; peer_<num_gpus;peer_++)
                        {
                            SizeT org_size = (peer_==0? graph_slice->frontier_queues[0].keys[frontier_attribute[0].selector^1].GetSize() : data_slice->keys_out[peer_].GetSize());
                            if (data_slice->out_length[peer_] > org_size)
                            {
                                printf("%d\t %lld\t %d\t keys_out   \t oversize :\t %d ->\t %d\n",
                                       thread_num, enactor_stats[0].iteration, peer_,
                                       org_size, data_slice->out_length[peer_]);fflush(stdout);
                                if (SIZE_CHECK)
                                {
                                    if (peer_==0) 
                                    {
                                        graph_slice->frontier_queues[0].keys[frontier_attribute[0].selector^1].EnsureSize(data_slice->out_length[0]);
                                    } else {
                                        data_slice -> keys_out[peer_].EnsureSize(data_slice->out_length[peer_]);
                                        for (int i=0;i<num_vertex_associate;i++)
                                        {
                                            data_slice->vertex_associate_out [peer_][i].EnsureSize(data_slice->out_length[peer_]);
                                            data_slice->vertex_associate_outs[peer_][i] = 
                                            data_slice->vertex_associate_out[peer_][i].GetPointer(util::DEVICE);
                                        }
                                        data_slice->vertex_associate_outs[peer_].Move(util::HOST, util::DEVICE, num_gpus, 0, streams[0]);
                                        for (int i=0;i<num_value__associate;i++)
                                        {
                                            data_slice->value__associate_out [peer_][i].EnsureSize(data_slice->out_length[peer_]);
                                            data_slice->value__associate_outs[peer_][i] = 
                                                data_slice->value__associate_out[peer_][i].GetPointer(util::DEVICE);
                                        }
                                        data_slice->value__associate_outs[peer_].Move(util::HOST, util::DEVICE, num_gpus, 0, streams[0]);
                                    }
                                } else {
                                    enactor_stats[0].retval = util::GRError(cudaErrorLaunchOutOfResources, "keys_out oversize", __FILE__, __LINE__);
                                    break;
                                }
                            }
                        }
                        if (enactor_stats[0].retval) break;
                    
                        for (int peer_=0;peer_<num_gpus;peer_++)
                            if (peer_==0) data_slice -> keys_outs[peer_] = graph_slice->frontier_queues[peer_].keys[frontier_attribute[0].selector^1].GetPointer(util::DEVICE);
                            else data_slice -> keys_outs[peer_] = data_slice -> keys_out[peer_].GetPointer(util::DEVICE);
                        data_slice->keys_outs.Move(util::HOST, util::DEVICE, num_gpus, 0, streams[0]);
                    
                        offset = 0; 
                        memcpy(&(data_slice -> make_out_array[offset]), 
                                 data_slice -> keys_markers         .GetPointer(util::HOST), 
                                  sizeof(SizeT*   ) * num_gpus);
                        offset += sizeof(SizeT*   ) * num_gpus ;
                        memcpy(&(data_slice -> make_out_array[offset]), 
                                 data_slice -> keys_outs            .GetPointer(util::HOST),
                                  sizeof(VertexId*) * num_gpus);
                        offset += sizeof(VertexId*) * num_gpus ;
                        memcpy(&(data_slice -> make_out_array[offset]), 
                                 data_slice -> vertex_associate_orgs.GetPointer(util::HOST),
                                  sizeof(VertexId*) * num_vertex_associate);
                        offset += sizeof(VertexId*) * num_vertex_associate ;
                        memcpy(&(data_slice -> make_out_array[offset]), 
                                 data_slice -> value__associate_orgs.GetPointer(util::HOST),
                                  sizeof(Value*   ) * num_value__associate);
                        offset += sizeof(Value*   ) * num_value__associate ;
                        for (int peer_=0; peer_<num_gpus; peer_++)
                        {    
                            memcpy(&(data_slice->make_out_array[offset]), 
                                     data_slice->vertex_associate_outs[peer_].GetPointer(util::HOST),
                                      sizeof(VertexId*) * num_vertex_associate);
                            offset += sizeof(VertexId*) * num_vertex_associate ;
                        }        
                        for (int peer_=0; peer_<num_gpus; peer_++)
                        {    
                            memcpy(&(data_slice->make_out_array[offset]), 
                                    data_slice->value__associate_outs[peer_].GetPointer(util::HOST),
                                      sizeof(Value*   ) * num_value__associate);
                            offset += sizeof(Value*   ) * num_value__associate ;
                        }                  
                        data_slice->make_out_array.Move(util::HOST, util::DEVICE, data_slice->make_out_array.GetSize(), 0, streams[0]);

                        Make_Out<VertexId, SizeT, Value, BfsProblem::MARK_PREDECESSORS?2:1, 0>
                            <<<grid_size, 256, sizeof(char)*data_slice->make_out_array.GetSize(), streams[0]>>> (
                            Total_Length,
                            num_gpus,
                            graph_slice->frontier_queues[num_gpus].keys[0].GetPointer(util::DEVICE),
                            graph_slice-> partition_table        .GetPointer(util::DEVICE),
                            graph_slice-> convertion_table       .GetPointer(util::DEVICE),
                            data_slice -> make_out_array         .GetSize(),
                            data_slice -> make_out_array         .GetPointer(util::DEVICE));
                            /*data_slice -> keys_markers           .GetPointer(util::DEVICE),
                            data_slice -> vertex_associate_orgs  .GetPointer(util::DEVICE),
                            data_slice -> value__associate_orgs  .GetPointer(util::DEVICE),
                            data_slice -> keys_outs              .GetPointer(util::DEVICE),
                            data_slice -> vertex_associate_outss .GetPointer(util::DEVICE),
                            data_slice -> value__associate_outss .GetPointer(util::DEVICE));
                            */
                        //if (enactor_stats[0].retval = util::GRError(cudaStreamSynchronize(streams[0]), "Make_Out error", __FILE__, __LINE__)) break;
                        /*if (!SIZE_CHECK) 
                        {
                            for (int peer_=0;peer_<num_gpus;peer_++)
                            cudaMemcpyAsync(&(data_slice->out_length[peer_]), 
                                data_slice->keys_marker[peer_].GetPointer(util::DEVICE) 
                                    + (Total_Length -1), 
                                sizeof(SizeT), cudaMemcpyDeviceToHost, streams[0]);
                        }*/
 
                        cudaStreamSynchronize(streams[0]);
                        frontier_attribute[0].selector^=1;
                        //if (enactor_stats[0].retval = util::GRError(cudaStreamSynchronize(streams[0]), "MemcpyAsync keys_marker error", __FILE__, __LINE__)) break;
                    } else {
                        for (int peer_=0;peer_<num_gpus;peer_++)
                            data_slice->out_length[peer_]=0;
                    }
                    for (int peer_=0;peer_<num_gpus;peer_++)
                        frontier_attribute[peer_].queue_length = data_slice->out_length[peer_];

                } else if (!All_Done<SizeT, DataSlice>(s_enactor_stats, s_frontier_attribute, s_data_slice, num_gpus)) {
                    if (enactor_stats[0].retval = work_progress[0].GetQueueLength(frontier_attribute[0].queue_index, frontier_attribute[0].queue_length, false, data_slice->streams[0])) break; 
                }
                enactor_stats[0].iteration++;
            }

            if (All_Done<SizeT, DataSlice>(s_enactor_stats, s_frontier_attribute, s_data_slice, num_gpus)) break;

            // Check if any of the frontiers overflowed due to redundant expansion
            /*for (int peer=0;peer<num_gpus;peer++)
            {
                bool overflowed = false;
                if (enactor_stats[peer].retval = work_progress[peer].CheckOverflow<SizeT>(overflowed)) break;
                if (overflowed) {
                    enactor_stats[peer].retval = util::GRError(cudaErrorInvalidConfiguration, "Frontier queue overflow. Please increase queue-sizing factor.",__FILE__, __LINE__);
                    break;
                }
            }*/
        } while(0);

        /*if (num_gpus >1) 
        {   
            if (break_clean) 
            {   
                util::cpu_mt::ReleaseBarrier(cpu_barrier,thread_num);
            }
        }*/
        //util::cpu_mt::PrintMessage("GPU BFS thread finished.", thread_num, enactor_stats->iteration);
        thread_data->stats=4;
        CUT_THREADEND;
    }

/**
 * @brief BFS problem enactor class.
 *
 * @tparam INSTRUMWENT Boolean type to show whether or not to collect per-CTA clock-count statistics
 */
template <typename _BfsProblem, bool _INSTRUMENT, bool _DEBUG, bool _SIZE_CHECK>
class BFSEnactor : public EnactorBase<typename _BfsProblem::SizeT, _DEBUG, _SIZE_CHECK>
{   
    _BfsProblem  *problem      ;
    ThreadSlice  *thread_slices;// = new ThreadSlice [this->num_gpus];
    CUTThread    *thread_Ids   ;// = new CUTThread   [this->num_gpus];

public:
    typedef _BfsProblem          BfsProblem;
    typedef typename BfsProblem::SizeT    SizeT   ;
    typedef typename BfsProblem::VertexId VertexId;
    typedef typename BfsProblem::Value    Value   ;
    static const bool INSTRUMENT = _INSTRUMENT;
    static const bool DEBUG      = _DEBUG;
    static const bool SIZE_CHECK = _SIZE_CHECK;

    // Methods

    /**
     * @brief BFSEnactor constructor
     */
    BFSEnactor(int num_gpus = 1, int* gpu_idx = NULL) :
        EnactorBase<SizeT, _DEBUG, _SIZE_CHECK>(VERTEX_FRONTIERS, num_gpus, gpu_idx)//,
    {
        //util::cpu_mt::PrintMessage("BFSEnactor() begin.");
        thread_slices = NULL;
        thread_Ids    = NULL;
        problem       = NULL;
        //util::cpu_mt::PrintMessage("BFSEnactor() end.");
    }

    /**
     * @brief BFSEnactor destructor
     */
    virtual ~BFSEnactor()
    {
        //util::cpu_mt::PrintMessage("~BFSEnactor() begin.");
        for (int gpu=0;gpu<this->num_gpus;gpu++)
        {
            util::SetDevice(this->gpu_idx[gpu]);
            if (BfsProblem::ENABLE_IDEMPOTENCE)
            {
                cudaUnbindTexture(gunrock::oprtr::filter::BitmaskTex<unsigned char>::ref);
            }
        }
        
        cutWaitForThreads(thread_Ids, this->num_gpus);
        delete[] thread_Ids   ; thread_Ids    = NULL;
        delete[] thread_slices; thread_slices = NULL;
        problem = NULL;
        //util::cpu_mt::PrintMessage("~BFSEnactor() end.");
    }

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief Obtain statistics about the last BFS search enacted.
     *
     * @param[out] total_queued Total queued elements in BFS kernel running.
     * @param[out] search_depth Search depth of BFS algorithm.
     * @param[out] avg_duty Average kernel running duty (kernel run time/kernel lifetime).
     */
    template <typename VertexId>
    void GetStatistics(
        long long &total_queued,
        VertexId  &search_depth,
        double    &avg_duty)
    {
        unsigned long long total_lifetimes=0;
        unsigned long long total_runtimes =0;
        total_queued = 0;
        search_depth = 0;
        for (int gpu=0;gpu<this->num_gpus;gpu++)
        {
            if (this->num_gpus!=1)
                if (util::SetDevice(this->gpu_idx[gpu])) return;
            cudaThreadSynchronize();

            total_queued += this->enactor_stats[gpu].total_queued;
            if (this->enactor_stats[gpu].iteration > search_depth) 
                search_depth = this->enactor_stats[gpu].iteration;
            total_lifetimes += this->enactor_stats[gpu].total_lifetimes;
            total_runtimes  += this->enactor_stats[gpu].total_runtimes;
        }
        avg_duty = (total_lifetimes >0) ?
            double(total_runtimes) / total_lifetimes : 0.0;
    }

    template<
        typename AdvanceKernelPolicy,
        typename FilterKernelPolicy>
    cudaError_t InitBFS(
        ContextPtr *context,
        BfsProblem *problem,
        int        max_grid_size = 0,
        bool       size_check    = true)
    {   
        cudaError_t retval = cudaSuccess;

        // Lazy initialization
        if (retval = EnactorBase<SizeT, DEBUG, SIZE_CHECK>::Init(problem,
                                       max_grid_size,
                                       AdvanceKernelPolicy::CTA_OCCUPANCY, 
                                       FilterKernelPolicy::CTA_OCCUPANCY)) return retval;
        
        this->problem = problem;
        thread_slices = new ThreadSlice [this->num_gpus];
        thread_Ids    = new CUTThread   [this->num_gpus];

        for (int gpu=0;gpu<this->num_gpus;gpu++)
        {
            if (retval = util::SetDevice(this->gpu_idx[gpu])) break;
            if (BfsProblem::ENABLE_IDEMPOTENCE) {
                int bytes = (problem->graph_slices[gpu]->nodes + 8 - 1) / 8;
                cudaChannelFormatDesc   bitmask_desc = cudaCreateChannelDesc<char>();
                gunrock::oprtr::filter::BitmaskTex<unsigned char>::ref.channelDesc = bitmask_desc;
                if (retval = util::GRError(cudaBindTexture(
                    0,
                    gunrock::oprtr::filter::BitmaskTex<unsigned char>::ref,//ts_bitmask[gpu],
                    problem->data_slices[gpu]->visited_mask.GetPointer(util::DEVICE),
                    bytes),
                    "BFSEnactor cudaBindTexture bitmask_tex_ref failed", __FILE__, __LINE__)) break;
            }
        }
        
        for (int gpu=0;gpu<this->num_gpus;gpu++)
        {
            thread_slices[gpu].thread_num    = gpu;
            thread_slices[gpu].problem       = (void*)problem;
            thread_slices[gpu].enactor       = (void*)this;
            thread_slices[gpu].context       = &(context[gpu*this->num_gpus]);
            thread_slices[gpu].stats         = -1;
            thread_slices[gpu].thread_Id = cutStartThread(
                (CUT_THREADROUTINE)&(BFSThread<
                    AdvanceKernelPolicy,FilterKernelPolicy, 
                    BFSEnactor<BfsProblem, INSTRUMENT, DEBUG, SIZE_CHECK> >),
                    (void*)&(thread_slices[gpu]));
            thread_Ids[gpu] = thread_slices[gpu].thread_Id;
        }

       return retval;
    }

    cudaError_t Reset()
    {
        return EnactorBase<SizeT, DEBUG, SIZE_CHECK>::Reset();
    }

    /** @} */

    /**
     * @brief Enacts a breadth-first search computing on the specified graph.
     *
     * @tparam EdgeMapPolicy Kernel policy for forward edge mapping.
     * @tparam FilterPolicy Kernel policy for filter.
     * @tparam BFSProblem BFS Problem type.
     *
     * @param[in] problem BFSProblem object.
     * @param[in] src Source node for BFS.
     * @param[in] max_grid_size Max grid size for BFS kernel calls.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    template<
        typename AdvanceKernelPolicy,
        typename FilterKernelPolicy>
    cudaError_t EnactBFS(
        VertexId    src)
    {
        clock_t      start_time = clock();
        cudaError_t  retval     = cudaSuccess;

        do {
            for (int gpu=0;gpu<this->num_gpus;gpu++)
            {
                if ((this->num_gpus ==1) || (gpu==this->problem->partition_tables[0][src]))
                     thread_slices[gpu].init_size=1;
                else thread_slices[gpu].init_size=0;
                this->frontier_attribute[gpu*this->num_gpus].queue_length = thread_slices[gpu].init_size;
            }
            
            for (int gpu=0; gpu< this->num_gpus; gpu++)
            {
                while (thread_slices[gpu].stats!=1) sleep(0);
                thread_slices[gpu].stats=2;
            }
            for (int gpu=0; gpu< this->num_gpus; gpu++)
            {
                while (thread_slices[gpu].stats!=4) sleep(0);
            }
 
            for (int gpu=0;gpu<this->num_gpus;gpu++)
            if (this->enactor_stats[gpu].retval!=cudaSuccess) {retval=this->enactor_stats[gpu].retval;break;}
        } while(0);

        if (this->DEBUG) printf("\nGPU BFS Done.\n");
        return retval;
    }

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief BFS Enact kernel entry.
     *
     * @tparam BFSProblem BFS Problem type. @see BFSProblem
     *
     * @param[in] problem Pointer to BFSProblem object.
     * @param[in] src Source node for BFS.
     * @param[in] max_grid_size Max grid size for BFS kernel calls.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    cudaError_t Enact(
        //ContextPtr  *context,
        //BFSProblem  *problem,
        VertexId    src)
        //int         max_grid_size = 0)
        //bool        size_check = true)
    {
        //util::cpu_mt::PrintMessage("BFSEnactor Enact() begin.");
        int min_sm_version = -1;
        for (int i=0;i<this->num_gpus;i++)
            if (min_sm_version == -1 || this->cuda_props[i].device_sm_version < min_sm_version)
                min_sm_version = this->cuda_props[i].device_sm_version;

        if (BfsProblem::ENABLE_IDEMPOTENCE) {
            //if (this->cuda_props.device_sm_version >= 300) {
            if (min_sm_version >= 300) {
                typedef gunrock::oprtr::filter::KernelPolicy<
                BfsProblem,                         // Problem data type
                300,                                // CUDA_ARCH
                INSTRUMENT,                         // INSTRUMENT
                0,                                  // SATURATION QUIT
                true,                               // DEQUEUE_PROBLEM_SIZE
                8,                                  // MIN_CTA_OCCUPANCY
                8,                                  // LOG_THREADS
                1,                                  // LOG_LOAD_VEC_SIZE
                0,                                  // LOG_LOADS_PER_TILE
                5,                                  // LOG_RAKING_THREADS
                5,                                  // END_BITMASK_CULL
                8>                                  // LOG_SCHEDULE_GRANULARITY
                    FilterKernelPolicy;

                typedef gunrock::oprtr::advance::KernelPolicy<
                    BfsProblem,                         // Problem data type
                    300,                                // CUDA_ARCH
                    INSTRUMENT,                         // INSTRUMENT
                    8,                                  // MIN_CTA_OCCUPANCY
                    10,                                  // LOG_THREADS
                    8,                                  // LOG_BLOCKS
                    32*128,                                  // LIGHT_EDGE_THRESHOLD (used for partitioned advance mode)
                    1,                                  // LOG_LOAD_VEC_SIZE
                    0,                                  // LOG_LOADS_PER_TILE
                    5,                                  // LOG_RAKING_THREADS
                    32,                            // WARP_GATHER_THRESHOLD
                    128 * 4,                            // CTA_GATHER_THRESHOLD
                    7,                                  // LOG_SCHEDULE_GRANULARITY
                    gunrock::oprtr::advance::LB>
                        AdvanceKernelPolicy;

                return EnactBFS<AdvanceKernelPolicy, FilterKernelPolicy>(
                        //context, problem, src, max_grid_size, size_check);
                        src);
            }
        } else {
                //if (this->cuda_props.device_sm_version >= 300) {
                if (min_sm_version >= 300) {
                typedef gunrock::oprtr::filter::KernelPolicy<
                    BfsProblem,                         // Problem data type
                300,                                // CUDA_ARCH
                INSTRUMENT,                         // INSTRUMENT
                0,                                  // SATURATION QUIT
                true,                               // DEQUEUE_PROBLEM_SIZE
                8,                                  // MIN_CTA_OCCUPANCY
                8,                                  // LOG_THREADS
                1,                                  // LOG_LOAD_VEC_SIZE
                0,                                  // LOG_LOADS_PER_TILE
                5,                                  // LOG_RAKING_THREADS
                5,                                  // END_BITMASK_CULL
                8>                                  // LOG_SCHEDULE_GRANULARITY
                    FilterKernelPolicy;

                typedef gunrock::oprtr::advance::KernelPolicy<
                    BfsProblem,                         // Problem data type
                    300,                                // CUDA_ARCH
                    INSTRUMENT,                         // INSTRUMENT
                    8,                                  // MIN_CTA_OCCUPANCY
                    10,                                  // LOG_THREADS
                    8,                                  // LOG_BLOCKS
                    32*128,                                  // LIGHT_EDGE_THRESHOLD (used for partitioned advance mode)
                    1,                                  // LOG_LOAD_VEC_SIZE
                    0,                                  // LOG_LOADS_PER_TILE
                    5,                                  // LOG_RAKING_THREADS
                    32,                            // WARP_GATHER_THRESHOLD
                    128 * 4,                            // CTA_GATHER_THRESHOLD
                    7,                                  // LOG_SCHEDULE_GRANULARITY
                    gunrock::oprtr::advance::LB>
                        AdvanceKernelPolicy;

                return EnactBFS<AdvanceKernelPolicy, FilterKernelPolicy>(
                        //context, problem, src, max_grid_size, size_check);
                        src);
            }
        }

        //to reduce compile time, get rid of other architecture for now
        //TODO: add all the kernelpolicy settings for all archs

        printf("Not yet tuned for this architecture\n");
        return cudaErrorInvalidDeviceFunction;
    }

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief BFS Enact kernel entry.
     *
     * @tparam BFSProblem BFS Problem type. @see BFSProblem
     *
     * @param[in] problem Pointer to BFSProblem object.
     * @param[in] src Source node for BFS.
     * @param[in] max_grid_size Max grid size for BFS kernel calls.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    cudaError_t Init(
        ContextPtr  *context,
        BfsProblem  *problem,
        //VertexId    src,
        int         max_grid_size = 0,
        bool        size_check = true)
    {
        //util::cpu_mt::PrintMessage("BFSEnactor Enact() begin.");
        int min_sm_version = -1;
        for (int i=0;i<this->num_gpus;i++)
            if (min_sm_version == -1 || this->cuda_props[i].device_sm_version < min_sm_version)
                min_sm_version = this->cuda_props[i].device_sm_version;

        if (BfsProblem::ENABLE_IDEMPOTENCE) {
            //if (this->cuda_props.device_sm_version >= 300) {
            if (min_sm_version >= 300) {
                typedef gunrock::oprtr::filter::KernelPolicy<
                BfsProblem,                         // Problem data type
                300,                                // CUDA_ARCH
                INSTRUMENT,                         // INSTRUMENT
                0,                                  // SATURATION QUIT
                true,                               // DEQUEUE_PROBLEM_SIZE
                8,                                  // MIN_CTA_OCCUPANCY
                8,                                  // LOG_THREADS
                1,                                  // LOG_LOAD_VEC_SIZE
                0,                                  // LOG_LOADS_PER_TILE
                5,                                  // LOG_RAKING_THREADS
                5,                                  // END_BITMASK_CULL
                8>                                  // LOG_SCHEDULE_GRANULARITY
                    FilterKernelPolicy;

                typedef gunrock::oprtr::advance::KernelPolicy<
                    BfsProblem,                         // Problem data type
                    300,                                // CUDA_ARCH
                    INSTRUMENT,                         // INSTRUMENT
                    8,                                  // MIN_CTA_OCCUPANCY
                    10,                                  // LOG_THREADS
                    8,                                  // LOG_BLOCKS
                    32*128,                                  // LIGHT_EDGE_THRESHOLD (used for partitioned advance mode)
                    1,                                  // LOG_LOAD_VEC_SIZE
                    0,                                  // LOG_LOADS_PER_TILE
                    5,                                  // LOG_RAKING_THREADS
                    32,                            // WARP_GATHER_THRESHOLD
                    128 * 4,                            // CTA_GATHER_THRESHOLD
                    7,                                  // LOG_SCHEDULE_GRANULARITY
                    gunrock::oprtr::advance::LB>
                        AdvanceKernelPolicy;

                return InitBFS<AdvanceKernelPolicy, FilterKernelPolicy>(
                        context, problem, max_grid_size,size_check);
            }
        } else {
                //if (this->cuda_props.device_sm_version >= 300) {
                if (min_sm_version >= 300) {
                typedef gunrock::oprtr::filter::KernelPolicy<
                    BfsProblem,                         // Problem data type
                300,                                // CUDA_ARCH
                INSTRUMENT,                         // INSTRUMENT
                0,                                  // SATURATION QUIT
                true,                               // DEQUEUE_PROBLEM_SIZE
                8,                                  // MIN_CTA_OCCUPANCY
                8,                                  // LOG_THREADS
                1,                                  // LOG_LOAD_VEC_SIZE
                0,                                  // LOG_LOADS_PER_TILE
                5,                                  // LOG_RAKING_THREADS
                5,                                  // END_BITMASK_CULL
                8>                                  // LOG_SCHEDULE_GRANULARITY
                    FilterKernelPolicy;

                typedef gunrock::oprtr::advance::KernelPolicy<
                    BfsProblem,                         // Problem data type
                    300,                                // CUDA_ARCH
                    INSTRUMENT,                         // INSTRUMENT
                    8,                                  // MIN_CTA_OCCUPANCY
                    10,                                  // LOG_THREADS
                    8,                                  // LOG_BLOCKS
                    32*128,                                  // LIGHT_EDGE_THRESHOLD (used for partitioned advance mode)
                    1,                                  // LOG_LOAD_VEC_SIZE
                    0,                                  // LOG_LOADS_PER_TILE
                    5,                                  // LOG_RAKING_THREADS
                    32,                            // WARP_GATHER_THRESHOLD
                    128 * 4,                            // CTA_GATHER_THRESHOLD
                    7,                                  // LOG_SCHEDULE_GRANULARITY
                    gunrock::oprtr::advance::LB>
                        AdvanceKernelPolicy;

                return InitBFS<AdvanceKernelPolicy, FilterKernelPolicy>(
                        context, problem, max_grid_size, size_check);
            }
        }

        //to reduce compile time, get rid of other architecture for now
        //TODO: add all the kernelpolicy settings for all archs

        printf("Not yet tuned for this architecture\n");
        return cudaErrorInvalidDeviceFunction;
    }


    /** @} */

};

} // namespace bfs
} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
