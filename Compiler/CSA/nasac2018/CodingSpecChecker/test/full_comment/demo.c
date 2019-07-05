typedef struct {
    char *name;
    unsigned char *buffer;
} device_t;

device_t *device_list;

/*
 * 拷贝设备缓冲区内容
 * 返回值：
 */
int copy_buffer(device_t *dst, device_t *src) {
    return 0;
}

// 测试设备是否在线
int test_device_online(device_t *dev) {
    return 0;
}

/*
 * 根据名称查找设备。
 */
device_t *get_device(const char *name) {
    return (void*)0;
}
