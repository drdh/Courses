#ifndef TESTCASE_SYNC
#define TESTCASE_SYNC

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/containers/string.hpp>

#include <cereal/archives/portable_binary.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/map.hpp>
#include <cereal/types/stack.hpp>
#include <cereal/types/string.hpp>

#include "Global_Data.h"

#define SHAREDSIZE 10485760

namespace ipc = boost::interprocess;
typedef ipc::allocator<char, boost::interprocess::managed_shared_memory::segment_manager> CharAllocator;
typedef ipc::basic_string<char, std::char_traits<char>, CharAllocator> shm_string;

class Sync {
public:

    void Save(Global_data *data);

    Global_data *Load();
};


#endif
