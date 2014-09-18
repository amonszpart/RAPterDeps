#ifndef __GF2_PEARL_HPP__
#define __GF2_PEARL_HPP__

#include <limits.h>

#include "Eigen/Dense"
#include "pearl/pearl.h"

#include "gco/GCoptimization.h"
#include "pcltools/util.hpp"
//#include "../ransacTest/include/primitives/houghLine.h"

namespace am
{
    template <class PointsT, class TLine>
    inline int
    Pearl::run(
            std::vector<int>          & labels
            , std::vector<TLine>      & lines
            , PointsT            const& cloud
            , std::vector<int>   const* indices
            , Params                    params
            , std::vector<std::vector<int> > *label_history
            , std::vector<std::vector<TLine> > *line_history
            )
    {
        int err = EXIT_SUCCESS;

        // prepare
        //lines.clear();
        labels.clear();

        // propose
        err += Pearl::propose( lines, cloud, indices, params );
        if ( err != EXIT_SUCCESS ) return err;
        std::cout << "lines.size(): " << lines.size() << std::endl;

        int iteration_id = 0;
        std::vector<int> prev_labels;
        do
        {
            std::cout << "\n[" << __func__ << "]: " << "iteration " << iteration_id << std::endl;
            prev_labels = labels;

            // expand
            err += Pearl::expand( labels
                                  , cloud
                                  , NULL
                                  , lines
                                  , /* gammasqr: */ params.gammasqr // 50*50
                                  , /*     beta: */ params.beta     // params.scale*100
                                  , params );
            if ( err != EXIT_SUCCESS ) return err;

            if ( label_history )
                label_history->emplace_back( labels );
            if ( line_history )
                line_history->emplace_back( lines );

            // refit
            err += Pearl::refit( lines, labels, *cloud, params ); // 25, 1000.f, 5e6f
            if ( err != EXIT_SUCCESS ) return err;

            if ( prev_labels.size() )
            {
                int label_diff = 0;
                for ( size_t i = 0; i != labels.size(); ++i )
                    label_diff += (labels[i] != prev_labels[i]);
                std::cout << "it[" << iteration_id << "] labeldiff: " << label_diff << std::endl;
            }

        } while ( /**/ (iteration_id++ != params.max_pearl_iterations)
                  &&   (!prev_labels.size() || !std::equal( labels.begin(), labels.end(), prev_labels.begin() )) );

        return err;
    }

    template <typename PointT, class TLine>
    inline int
    Pearl::propose( std::vector<TLine>                            & lines
                    , boost::shared_ptr<pcl::PointCloud<PointT> >   cloud
                    , std::vector<int>                       const* indices
                    , Params                                 const& params
                    )
    {
        typedef typename TLine::Scalar Scalar;

        if ( indices ) { std::cerr << __PRETTY_FUNCTION__ << "]: indices must be NULL, not implemented yet..." << std::endl; return EXIT_FAILURE; }

        // get neighbourhoods
        std::vector<std::vector<int   > > neighs;
        std::vector<std::vector<Scalar> > sqr_dists;
        smartgeometry::getNeighbourhoodIndices( neighs
                                                , cloud
                                                , indices
                                                , &sqr_dists
                                                , params.max_neighbourhood_size   // 5
                                                , params.scale // 0.02f
                                                , /* soft_radius: */ true
                                                );

        // every point proposes primitive[s] using its neighbourhood
        for ( size_t pid = 0; pid != neighs.size(); ++pid )
        {
#if 1
            // can't fit a line to 0 or 1 points
            if ( neighs[pid].size() < 2 ) continue;

            Eigen::Matrix<Scalar,TLine::Dim,1> line;
            int err = smartgeometry::geometry::fitLinearPrimitive( /*           output: */ line
                                                                   , /*         points: */ *cloud
                                                                   , /*          scale: */ params.scale
                                                                   , /*        indices: */ &(neighs[pid])
                                                                   , /*    refit times: */ 2
                                                                   , /* use input line: */ false
                                                                   );

            if ( err == EXIT_SUCCESS )
                lines.emplace_back( TLine(line) );
#else
            // skip, if no neighbours found this point won't contribute a primitive for now
            if ( neighs[pid].size() < 2 )  continue;

            // compute neighbourhood cov matrix
            Eigen::Matrix<Scalar,3,3> cov;
            smartgeometry::computeCovarianceMatrix<pcl::PointCloud<PointT>,float>( cov, *cloud, &(neighs[pid]), NULL, NULL );
            // solve for neighbourhood biggest eigen value
            Eigen::SelfAdjointEigenSolver< Eigen::Matrix<Scalar, 3, 3> > es;
            es.compute( cov );
            // get eigen vector for biggest eigen value
            const int max_eig_val_id = std::distance( es.eigenvalues().data(), std::max_element( es.eigenvalues().data(), es.eigenvalues().data()+3 ) );
            Eigen::Matrix<Scalar,3,1> eig2 = es.eigenvectors().col( max_eig_val_id ).normalized();
            // compute line direction perpendicular to eigen vector
            Eigen::Matrix<Scalar,3,1> p0 = cloud->at(pid).getVector3fMap();                                      // point on line

            lines.emplace_back( TLine(p0, eig2) );
#endif
        }

        return EXIT_SUCCESS;
    }

    template <typename PointT, class TLine, typename Scalar >
    inline int
    Pearl::expand(
            std::vector<int>          & labels
            , boost::shared_ptr<pcl::PointCloud<PointT> > cloud
            , std::vector<int>   const* indices
            , std::vector<TLine> const& lines
            , float              const  gammasqr
            , float              const  beta
            , Params             const& params )
    {
        if ( indices ) { std::cerr << __PRETTY_FUNCTION__ << "]: indices must be NULL, not implemented yet..." << std::endl; return EXIT_FAILURE; }

        const int num_pixels = indices ? indices->size() : cloud->size();
        const int num_labels = lines.size();
        labels.resize( num_pixels );

        // first set up the array for data costs
        const float intMultSqr = params.int_mult * params.int_mult;
        std::cout << "intMult: " << params.int_mult << std::endl;
        Scalar *data = new Scalar[ num_pixels * num_labels ];
        int data_zero_cnt = 0, data_max_cnt = 0;
        for ( int pid = 0; pid != num_pixels; ++pid )
            for ( int line_id = 0; line_id != num_labels; ++line_id )
            {
                float dist = lines[line_id].getDistance( /*     point: */ cloud->at( indices ? (*indices)[pid] : pid ).getVector3fMap() );
                // std::cout << "dist: " << dist;
                dist *= dist * intMultSqr;
                // std::cout << ", data: " << dist << std::endl;

                if ( (dist != dist) || (dist < 0) || (dist > params.data_max) )
                {
                    dist = params.data_max;
                    ++data_max_cnt;
                }
                else if (dist == 0)
                    ++data_zero_cnt;

                data[ pid * num_labels + line_id ] = dist;
            }
        std::cout << "Zero datacosts: " << data_zero_cnt << "/" << num_pixels << "(" << Scalar(data_zero_cnt)/num_pixels*Scalar(100.) << "%)"
                  << ", capped datacost: " << data_max_cnt << "/" << num_pixels << "(" << Scalar(data_max_cnt)/num_pixels*Scalar(100.) << "%)\n";

        // next set up the array for smooth costs
        //const int smooth_2 = params.lambdas(2)/2;
        Scalar *smooth = new Scalar[ num_labels * num_labels ];
        for ( int l1 = 0; l1 < num_labels; l1++ )
            for (int l2 = 0; l2 < num_labels; l2++ )
                smooth[l1+l2*num_labels] = /*smooth_2 * */ (l1 != l2); // dirac/potts

        std::vector<std::vector<int  > > neighs;
        std::vector<std::vector<float> > sqr_dists;
        smartgeometry::getNeighbourhoodIndices( neighs
                                                , cloud
                                                , indices
                                                , &sqr_dists
                                                , params.max_neighbourhood_size   // 10
                                                , params.scale // 0.01f
                                                );
        try
        {
            GCoptimizationGeneralGraph *gc = new GCoptimizationGeneralGraph(num_pixels,num_labels);
            gc->setDataCost  ( data   );
            gc->setSmoothCost( smooth );
            gc->setLabelCost ( beta   );

            // set neighbourhoods
            std::vector<int> neighvals; neighvals.reserve( neighs.size() * 15 );
            for ( size_t pid = 0; pid != neighs.size(); ++pid )
                for ( size_t pid2 = 1; pid2 != neighs[pid].size(); ++pid2 ) // don't count own
                {
                    float distsqr  = sqr_dists[pid][pid2] * params.int_mult * params.int_mult;
                    long  neighval = params.lambdas(2) * exp( -1.f * distsqr / gammasqr);

                    if ( neighval > (long)INT_MAX )
                        std::cerr << "neighbours exceeds limits: " << neighval << std::endl;

                    if ( neighval < 0 )
                        std::cerr << "neighval: " << neighval << std::endl;

                    gc->setNeighbors( pid, neighs[pid][pid2], neighval );
                    neighvals.push_back( neighval );
                }

            // debug
            {
                int sum = std::accumulate( neighvals.begin(), neighvals.end(), 0 );
                std::cout << "neighval avg: " << sum / (float)neighvals.size();
                std::sort( neighvals.begin(), neighvals.end() );
                std::cout << ", median: " << neighvals[ neighvals.size() / 2 ];
                int n_min = neighvals[0];
                std::cout << ", neighval_min: " << n_min << ", neighval_max: " << neighvals.back() << std::endl;
            }

            if ( neighvals[ neighvals.size() / 2 ] > 0 )
            {
                std::cout << params.beta << ", " << params.lambdas(2) << ", " << params.pottsweight;
                if ( (params.lambdas(2) * params.pottsweight < 1e10f) )
                {

                    printf("\tBefore optimization energy is %lld", gc->compute_energy() );
                    gc->expansion( 10 );// run expansion for 2 iterations. For swap use gc->swap(num_iterations);
                    printf("\tAfter optimization energy is %lld\n",gc->compute_energy());

                    // copy output
                    for ( int pid = 0; pid != num_pixels; ++pid )
                    {
                        labels[ pid ] = gc->whatLabel( pid );
                    }
                }
                else
                {
                    std::cout << "skipping " << std::endl;
                    for ( int pid = 0; pid != num_pixels; ++pid )
                    {
                        labels[ pid ] = 0;
                    }
                }
            }
            else
                std::cerr << "[" << __func__ << "]: " << "effective pairwise median !> 0, so not running" << std::endl;

            // cleanup
            if ( gc     ) { delete gc; gc = NULL; }
        }
        catch ( GCException e )
        {
            e.Report();
        }
        catch ( std::exception &e )
        {
            std::cerr << "[" << __func__ << "]: " << e.what() << std::endl;
        }

        // cleanup
        if ( data   ) { delete [] data  ; data   = NULL; }
        if ( smooth ) { delete [] smooth; smooth = NULL; }

        return EXIT_SUCCESS;
    }

    template <typename PointsT, class TLine> inline int
    Pearl::refit( std::vector<TLine>        &lines
                  , std::vector<int> const& labels
                  , PointsT          const& cloud
                  , Params           const& params
                  //, std::vector<int> const* indices // TODO: this is not used...
                  )
    {
        using std::vector;
        //typedef typename TLine::Scalar Scalar;

        // assign points to lines
        vector<vector<int> > lines_points( lines.size() );
        for ( size_t pid = 0; pid != labels.size(); ++pid )
            lines_points[ labels[pid] ].push_back( pid );

        for ( size_t line_id = 0; line_id != lines.size(); ++line_id )
        {
#if 1
            if ( lines_points[line_id].size() > 1 )
                smartgeometry::geometry::fitLinearPrimitive( /*           output: */ lines[line_id].coeffs()
                                                             , /*         points: */ cloud
                                                             , /*          scale: */ params.scale
                                                             , /*        indices: */ &(lines_points[line_id])
                                                             , /*    refit times: */ 2
                                                             , /* use input line: */ true );
#else

            if ( lines_points[line_id].size() < 2 ) continue;

            // calculate weights
            Eigen::Matrix<Scalar,-1,1> weights_matrix( lines_points[line_id].size(), 1 );
            weights_matrix.setZero();

            std::vector<Scalar> weights( lines_points[line_id].size(), 1.f );
            for ( size_t member_id = 0; member_id != lines_points[line_id].size(); ++member_id )
            {
                weights[member_id] = lines[line_id].distanceToPoint( cloud[ /*indices_arg?indices_arg[ : */ lines_points[line_id][member_id] ].getVector4fMap(), false );
                weights[member_id] *= params.scale;
                weights[member_id] *= weights[member_id];
                weights_matrix(member_id) = weights[member_id];
            }
            Eigen::Matrix<Scalar,-1,1> demean = weights_matrix.rowwise() - weights_matrix.colwise().mean();
            Eigen::Matrix<Scalar,-1,1> squared = demean.array().square();
            Scalar stddev = sqrt(squared.sum() / weights_matrix.rows());
            std::cout << "weights_matrix.stddev(): " << stddev << std::endl;
            demean /= stddev * stddev;
            std::cout << "demean.maxCoeff(): " << demean.maxCoeff() << std::endl;

            float max_weight = *std::max_element( weights.begin(), weights.end() );
            if ( params.debug ) std::cout << "max_weight: " << max_weight << std::endl;

            std::for_each( weights.begin(), weights.end(), [&weights,&max_weight]( float &weight ){ weight = max_weight - weight; } );
            if ( params.debug ) {std::cout<<"weights:";for(size_t vi=0;vi!=weights.size();++vi)std::cout<<((weights[vi]==0.f)?"\t!!!":"") << weights[vi]<<" ";std::cout << "\n";}

            // compute neighbourhood cov matrix
            Eigen::Matrix<Scalar,4,1> centroid;
            smartgeometry::computeCentroid<PointsT,Scalar>( centroid, cloud, &(lines_points[line_id]) );
            Eigen::Matrix<Scalar,3,3> cov;
            smartgeometry::computeCovarianceMatrix<PointsT,Scalar>( cov, cloud, &(lines_points[line_id]), &centroid, &weights );

            // solve for neighbourhood biggest eigen value
            Eigen::SelfAdjointEigenSolver< Eigen::Matrix<Scalar, 3, 3> > es;
            es.compute( cov );

            // get eigen vector for biggest eigen value
            const int max_eig_val_id = std::distance( es.eigenvalues().data(), std::max_element( es.eigenvalues().data(), es.eigenvalues().data()+3 ) );
            Eigen::Matrix<Scalar,3,1> eig2 = es.eigenvectors().col( max_eig_val_id ).normalized();

            if ( params.debug ) std::cout << "anglediff[" << line_id << "/" << lines.size() << "]: " << acos( lines[line_id].dir().dot( eig2 ) ) * 180.f/M_PI << std::endl;

            lines[line_id] = TLine( centroid.template head<3>(), eig2 );
#endif
        }

        return EXIT_SUCCESS;
    }


    inline int
    Pearl::getActiveLabelsCount( std::vector<int>         const& labels
                                 , std::vector<unsigned>       * active_labels_arg )
    {
        std::vector<unsigned> active_labels;
        for ( size_t pid = 0; pid != labels.size(); ++pid )
        {
            if ( static_cast<int>(active_labels.size()) <= labels[pid] )
                active_labels.resize( labels[pid]+1, 0 );
            active_labels[ labels[pid] ] = 1;
        }

        if ( active_labels_arg )
            *active_labels_arg = active_labels;

        return std::accumulate( active_labels.begin(), active_labels.end(), 0 );
    }
} // nsam

#endif // __GF2_PEARL_HPP__
