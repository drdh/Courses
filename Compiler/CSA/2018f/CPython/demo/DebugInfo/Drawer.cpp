
#include "Drawer.h"

void DrawerGraphviz::draw(Global_data data) {
    std::map<std::string, Agnode_t *> node_list;//函数mangle名到节点指针的映射
    std::map<std::string, int> edge_list;//边名(mangle组合)到调用次数的映射
    static GVC_t *gvc;
    if (!gvc)
        gvc = gvContext();
    Agraph_t *g;
    char tmp[] = "g";
    g = agopen(tmp, Agstrictdirected, NIL(Agdisc_t *));
    agsafeset(g, "bgcolor", "transparent", "");

    std::string start_m_name;
    std::string end_m_name;
    std::string start_d_name;
    std::string end_d_name;
    std::string edge_name;
    std::map<std::string, Function> func_list = data.get_function_list();
    for (auto func_iter = func_list.begin(); func_iter != func_list.end(); func_iter++) {
        Agnode_t *func;
        start_m_name = func_iter->second.getmangle_name();
        start_d_name = func_iter->second.getdemangle_name();
        char *func_name = new char[start_d_name.length() + 1];
        strcpy(func_name, start_d_name.c_str());
        func = agnode(g, func_name, 1);// 根据调用者确定起点
        char max_time[40];
        sprintf(max_time, "%s\nmax time:%fms", start_d_name.c_str(), 1000 * func_iter->second.getmax_runtime());
        agsafeset(func, "label", max_time, "");
        if(func_iter->second.getlanguage() == CPP) {
            //agsafeset(func, "color", "lightblue", "");
            agsafeset(func, "color", "#CBD9EF", "");
            agsafeset(func, "style", "filled", "");
        }else if(func_iter->second.getlanguage() == PYTHON){
            //agsafeset(func, "color", ".7 .3 1.0", "");
            agsafeset(func, "color", "lightblue", "");
            agsafeset(func, "style", "filled", "");
        }else{
            agsafeset(func, "color", "#416D94", "");
            agsafeset(func, "style", "filled", "");
        }
        node_list[start_m_name] = func;

        std::vector<Call> callee = func_iter->second.getcallee_list();
        for (auto call_iter = callee.begin(); call_iter != callee.end(); call_iter++) {
            Agedge_t *e;
            Agnode_t *s;
            end_m_name = call_iter->mangle_name;
            end_d_name = call_iter->demangle_name;
            if (node_list.find(end_m_name) == node_list.end()) {
                char *s_name = new char[end_d_name.length() + 1];
                strcpy(s_name, end_d_name.c_str());
                s = agnode(g, s_name, 1);// 根据被调用者确定终点
                node_list[end_m_name] = s;
                delete[] s_name;
            }
            s = node_list[end_m_name];
            edge_name = start_m_name + "@" + end_m_name;
            if (edge_list.find(edge_name) == edge_list.end()) {
                edge_list[edge_name] = 1;
            } else {
                edge_list[edge_name] += 1;
            }
            e = agedge(g, func, s, NULL, 1);
            std::string num_str = std::to_string(edge_list[edge_name]);
            char *num_cstr = new char[num_str.length() + 1];
            strcpy(num_cstr, num_str.c_str());
            agsafeset(e, "label", num_cstr, "");
	    agsafeset(e, "color", "white", "");
            agsafeset(e, "fontcolor", "white", "");
        }
        delete[] func_name;
    }
    gvLayout(gvc, g, "dot");
    gvRender(gvc, g, "png", fopen("graph.png", "w"));
    gvFreeLayout(gvc, g);
    agclose(g);
}

void DrawerPrint::draw(Global_data data) {
    std::cout << "-----------------------------" << std::endl;
    std::map<std::string, Function> func_list = data.get_function_list();
    std::vector<Call> call_list;
    for (auto func_iter = func_list.begin(); func_iter != func_list.end(); func_iter++) {
        Function func = func_iter->second;
        std::cout << "name: " << func.getdemangle_name() << std::endl;
        std::cout << "call_num: " << func.getcall_num() << std::endl;
        std::cout << "type: " << (int) func.gettype() << std::endl;
        std::cout << "language: " << (int) func.getlanguage() << std::endl;
        std::cout << "total_runtime: " << func.gettotal_runtime() << std::endl;
        std::cout << "min_runtime: " << func.getmin_runtime() << std::endl;
        std::cout << "max_runtime: " << func.getmax_runtime() << std::endl;
        std::cout << "callers: " << std::endl;

        call_list = func.getcaller_list();
        for (auto call_iter = call_list.begin(); call_iter != call_list.end(); call_iter++) {
            std::cout << call_iter->demangle_name << ":\tbegintime:" << call_iter->begin_time <<
                      "\tendtime:" << call_iter->end_time << "\texectime:" << call_iter->exec_time << std::endl;
        }
        std::cout << "callees: " << std::endl;
        call_list = func.getcallee_list();
        for (auto call_iter = call_list.begin(); call_iter != call_list.end(); call_iter++) {
            std::cout << call_iter->demangle_name << ":\tbegintime:" << call_iter->begin_time <<
                      "\tendtime:" << call_iter->end_time << "\texectime:" << call_iter->exec_time << std::endl;
        }
        std::cout << "---------------------------------" << std::endl;
    }
}

DrawerBase *DrawerPick::get(DRAWER type) {
    switch (type) {
        case Graphviz:
            return new DrawerGraphviz();
            break;
        case Print:
            return new DrawerPrint();
            break;
        default:
            return NULL;
            break;
    }
}
