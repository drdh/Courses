#include "Sync.h"

void Sync::Save(Global_data *data) {
    ipc::shared_memory_object::remove("Debug");
    std::stringstream ss(std::ios::binary | std::ios::in | std::ios::out);
    cereal::PortableBinaryOutputArchive oarchive(ss);
    boost::interprocess::managed_shared_memory shm(boost::interprocess::open_or_create, "Debug", SHAREDSIZE);
    oarchive(*data);

    CharAllocator alloc(shm.get_segment_manager());
    shm_string b(alloc);

    std::string tmp = ss.str();
    b.assign(tmp.begin(), tmp.end());

    shm.construct<shm_string>("global")(b, shm.get_segment_manager());
    // delete data;
}

Global_data* Sync::Load() {
    std::stringstream ss(std::ios::binary | std::ios::in | std::ios::out);
    boost::interprocess::managed_shared_memory shm(boost::interprocess::open_or_create, "Debug", SHAREDSIZE);
    shm_string *info = shm.find<shm_string>("global").first;
    Global_data *data = new Global_data;
    if (info == NULL) {
        return data;
    }
    std::string i;
    i.assign(info->begin(), info->end());
    ss << i;
    cereal::PortableBinaryInputArchive iarchive(ss);
    iarchive(*data);
    ipc::shared_memory_object::remove("Debug");
    return data;
}
