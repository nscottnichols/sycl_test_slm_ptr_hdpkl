#include <iostream>
#include <cassert>
#include <CL/sycl.hpp>
#ifndef SUB_GROUP_SIZE
    #define SUB_GROUP_SIZE 32
#endif
#ifndef WORK_GROUP_SIZE
    #define WORK_GROUP_SIZE 32
#endif
#ifndef SLM_SIZE
    #define SLM_SIZE 3
#endif

int usage(char* argv0, int ret = 1) {
    std::cout << "Usage: " << argv0
              << " [-h] [-N]" << std::endl << std::endl;
              //<< " [-h] [-N] [-L]" << std::endl << std::endl;
    std::cout << "Optional arguments:"                                                                          << std::endl;
    std::cout << "  -h, --help                          shows help message and exits"                           << std::endl;
    std::cout << "  -N, --number_of_elements            number of elements in arrays (default: 8)"              << std::endl;
    //std::cout << "  -L, --number_of_kernel_launches     number of dot product kernel launches (default: 1)"     << std::endl;
    //std::cout << "  -R, --number_of_repititions         number of times to repeat kernel launches (default: 1)" << std::endl;
    return ret;
}

int main(int argc, char **argv) {
    //Parse arguments
    size_t N = 8;
    //size_t L = 1;
    //size_t R = 1;
    for (int argn = 1; argn < argc; ++argn) {
        std::string arg = argv[argn];
        if ((arg == "--help") || (arg == "-h")) {
            return usage(argv[0]);
        } else if ((arg == "--number_of_elements") || (arg == "-N")) {
            N = static_cast<size_t>(strtoul(argv[argn + 1], NULL, 0));
            argn++;
        }
        //} else if ((arg == "--number_of_kernel_launches") || (arg == "-L")) {
        //    L = static_cast<size_t>(strtoul(argv[argn + 1], NULL, 0));
        //    argn++;
        //} else if ((arg == "--number_of_repititions") || (arg == "-R")) {
        //    R = static_cast<size_t>(strtoul(argv[argn + 1], NULL, 0));
        //    argn++;
        //}
    }

    std::cout << "N = " << N << std::endl;

    //Setup device
    auto devices = sycl::device::get_devices();
    sycl::queue q = sycl::queue(devices[0]);
    
    //Test for valid subgroup size
    //auto sg_sizes = q.get_device().get_info<sycl::info::device::sub_group_sizes>();
    //if (std::none_of(sg_sizes.cbegin(), sg_sizes.cend(), [](auto i) { return i  == SUB_GROUP_SIZE; })) {
    //    std::stringstream ss;
    //    ss << "Invalid SUB_GROUP_SIZE. Please select from: ";
    //    for (auto it = sg_sizes.cbegin(); it != sg_sizes.cend(); it++) {
    //        if (it != sg_sizes.begin()) {
    //            ss << " ";
    //        }
    //        ss << *it;
    //    }
    //    throw std::runtime_error(ss.str());
    //}

    //test unconventional slm
    auto h_a  = (double*) malloc(sizeof(double)*N); 
    auto h_c  = (double*) malloc(sizeof(double)*L); 

    //set host arrays
    for (size_t i=0; i<SLM_SIZE; i++) {
        h_a[i] = static_cast<double>(i);
    }

    //Set device memory so reads are coming from separate memory locations
    auto _usm = sycl::malloc_device<double>(SLM_SIZE, q); 
    auto d_c  = sycl::malloc_device<double>(N       , q); 

    q.memcpy(_usm, h_a, sizeof(double)*SLM_SIZE);
    q.wait();

    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for_work_group(sycl::range<1>{grid_size}, sycl::range<1>{WORK_GROUP_SIZE}, ([=](sycl::group<1> work_group) {
            auto _slm = usm_ptr;
            work_group.parallel_for_work_item([&](sycl::h_item<1> it) {
                size_t global_idx = it.get_global_id(0);
                if (global_idx < N) {
                    double pvt_mem_var[SLM_SIZE + 1];

                    //set data in private memory
                    pvt_mem_var[0] = static_cast<double>(global_idx);
                    for (size_t i = 0; i < SLM_SIZE; i++) {
                        pvt_mem_var[i + 1] = _slm[i];
                    }

                    //set USM var d_c as sum over pvt_mem_var
                    d_c[global_idx] = 0.0;
                    for (size_t i = 0; i < SLM_SIZE + 1; i++) {
                        d_c[global_idx] += pvt_mem_var[i];
                    }
                }
            });

            //implicit work group barrier

            work_group.parallel_for_work_item([&](sycl::h_item<1> it) {
                size_t global_idx = it.get_global_id(0);
                size_t local_idx = it.get_local_id(0);

                //mess with data in slm
                if (local_idx == 0) {
                    _slm[0] = 99999999.99999999
                }
            });

            //implicit work group barrier

            work_group.parallel_for_work_item([&](sycl::h_item<1> it) {
                size_t global_idx = it.get_global_id(0);
                if (global_idx < N) {
                    size_t local_idx = it.get_local_id(0);

                    // sum from SLM if group leader, otherwise sume from USM
                    if (local_idx == 0) {
                        for (size_t i = 0; i < SLM_SIZE; i++) {
                            d_c[global_idx] += _slm[i];
                        }
                    } else {
                        for (size_t i = 0; i < SLM_SIZE; i++) {
                            d_c[global_idx] += _usm[i];
                        }
                    }
                }
            });
        }));
    });
    q.wait();

    q.memcpy(h_c, d_c, sizeof(double)*N).wait();

    sycl::free(_usm, q);
    sycl::free(d_c , q);

    for (size_t i=0; i < N; i++) {
        std::cout << "h_c[" << i << "]:  " << h_c[i]  << std::endl; 
    }

    free(h_a );
    free(h_c );
}
