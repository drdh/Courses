
//`define FIFO_LRU  //0:FIFO  1:LRU
module cache #(
    parameter  LINE_ADDR_LEN = 3, // line内地�?长度，决定了每个line具有2^3个word
    parameter  SET_ADDR_LEN  = 3, // 组地�?长度，决定了�?共有2^3=8�?
    parameter  TAG_ADDR_LEN  = 5, // tag长度
    parameter  WAY_CNT       = 3  // 组相连度，决定了每组中有多少路line，这里是直接映射型cache，因此该参数没用�?
)(
    input  clk, rst,
    output miss,               // 对CPU发出的miss信号
    input  [31:0] addr,        // 读写请求地址
    input  rd_req,             // 读请求信�?
    output reg [31:0] rd_data, // 读出的数据，�?次读�?个word
    input  wr_req,             // 写请求信�?
    input  [31:0] wr_data      // 要写入的数据，一次写�?个word
);
//主存每次读一个line的数�?
localparam MEM_ADDR_LEN    = TAG_ADDR_LEN + SET_ADDR_LEN ; // 计算主存地址长度 MEM_ADDR_LEN，主存大�?=2^MEM_ADDR_LEN个word
localparam UNUSED_ADDR_LEN = 32 - MEM_ADDR_LEN - 2 -LINE_ADDR_LEN ;       // 计算未使用的地址的长�?
//此处之前没有-LINE_ADDR_LEN, 但是很明显需要加�?

localparam LINE_SIZE       = 1 << LINE_ADDR_LEN  ;         // 计算 line �? word 的数量，�? 2^LINE_ADDR_LEN 个word �? line
localparam SET_SIZE        = 1 << SET_ADDR_LEN   ;         // 计算�?共有多少组，�? 2^SET_ADDR_LEN 个组

reg [            31:0] cache_mem    [SET_SIZE][WAY_CNT][LINE_SIZE]; // SET_SIZE个line，每个line有LINE_SIZE个word
reg [TAG_ADDR_LEN-1:0] cache_tags   [SET_SIZE][WAY_CNT];            // SET_SIZE个TAG
reg                    valid        [SET_SIZE][WAY_CNT];            // SET_SIZE个valid(有效�?)
reg                    dirty        [SET_SIZE][WAY_CNT];            // SET_SIZE个dirty(脏位)

wire [              2-1 :0]   word_addr;                   // 将输入地�?addr拆分成这5个部�?
wire [  LINE_ADDR_LEN-1 :0]   line_addr;
wire [   SET_ADDR_LEN-1 :0]    set_addr;
wire [   TAG_ADDR_LEN-1 :0]    tag_addr;
wire [UNUSED_ADDR_LEN-1 :0] unused_addr;

enum  {IDLE, SWAP_OUT, SWAP_IN, SWAP_IN_OK} cache_stat;    // cache 状�?�机的状态定�?
                                                           // IDLE代表就绪，SWAP_OUT代表正在换出，SWAP_IN代表正在换入，SWAP_IN_OK代表换入后进行一周期的写入cache操作�?

reg [   SET_ADDR_LEN-1 :0] mem_rd_set_addr = 0;
reg [   TAG_ADDR_LEN-1 :0] mem_rd_tag_addr = 0;
wire[   MEM_ADDR_LEN-1 :0] mem_rd_addr = {mem_rd_tag_addr, mem_rd_set_addr};
reg [   MEM_ADDR_LEN-1 :0] mem_wr_addr = 0;

reg  [31:0] mem_wr_line [LINE_SIZE]; //mem操作的单位为�?个line长度
wire [31:0] mem_rd_line [LINE_SIZE];

wire mem_gnt;      // 主存响应读写的握手信�?

assign {unused_addr, tag_addr, set_addr, line_addr, word_addr} = addr;  // 拆分 32bit ADDR


reg cache_hit = 1'b0;
reg [WAY_CNT-1:0]hit_way;

always @ (*) begin              // 判断 输入的address 是否�? cache 中命�?
    cache_hit=1'b0;
	for( integer i=0;i<WAY_CNT;i++)begin
		if(valid[set_addr][i] && cache_tags[set_addr][i]==tag_addr)begin // 如果 cache line有效，并且tag与输入地�?中的tag相等，则命中
			cache_hit=1'b1;
			hit_way=i;
		end
	end
end

`ifndef FIFO_LRU
reg [WAY_CNT-1:0]FIFO_pointer[SET_SIZE];//对于FIFO,每一组都有一个写入的目标,依次递增
`else
reg [WAY_CNT-1:0]LRU_stack[SET_SIZE][WAY_CNT];//对于LRU,保存�?个栈,每次access�?个，就把它换到尾部，每次写入都是头部
reg [WAY_CNT-1:0]LRU_find;
`endif
reg [WAY_CNT-1:0]which_way;//选一路写入cache(或�?�该路需要先换出)

//1.找到which_way,为了统一格式，命中时，直接赋值给which_way
//2.更新FIFO_pointer,LRU_stack

always@(posedge (rd_req|wr_req))begin
    if(cache_hit)begin//命中时�??
        which_way=hit_way;
		`ifdef FIFO_LRU
		//if(FIFO_LRU==1'b1)begin//LRU时，�?要将这个access的路的需要放到队�?
            for( integer i=0;i<WAY_CNT;i++)begin
                if(LRU_stack[set_addr][i]==hit_way)begin
                    LRU_find=i;//表示命中的在第LRU_find位置�?
                end
            end
            for( integer i=1;i<WAY_CNT;i++)begin
				if(i>LRU_find)begin//后面的前�?
					LRU_stack[set_addr][i-1]=LRU_stack[set_addr][i];
				end
            end
            LRU_stack[set_addr][WAY_CNT-1]=which_way;//放到队尾
        //end
		`endif
        //FIFO不需要处�?
    end else begin//未命中时
		`ifndef FIFO_LRU
        //if(FIFO_LRU==1'b0)begin //FIFO
            which_way=FIFO_pointer[set_addr];
            if(which_way==WAY_CNT-1)begin
                FIFO_pointer[set_addr]<=0;
            end else begin
                FIFO_pointer[set_addr]+=1;
            end
		`else
        //end else begin //LRU
            which_way=LRU_stack[set_addr][0];
            for( integer i=1;i<WAY_CNT;i++)begin
                LRU_stack[set_addr][i-1]=LRU_stack[set_addr][i];
            end
            LRU_stack[set_addr][WAY_CNT-1]=which_way;
        //end
		`endif
    end
end

always @ (posedge clk or posedge rst) begin     // ?? cache ???
    if(rst) begin
        cache_stat <= IDLE;
        for( integer i=0; i<SET_SIZE; i++) begin
			`ifndef FIFO_LRU
            //if(FIFO_LRU==1'b0)begin
                FIFO_pointer[i]=0;//FIFO指针，初始化时，指向0, 作为下一个替换的目标
            //end
			`endif
            for( integer j=0;j<WAY_CNT;j++)begin
                dirty[i][j] = 1'b0;
                valid[i][j] = 1'b0;
				`ifdef FIFO_LRU
                //if(FIFO_LRU==1'b1)begin
                    LRU_stack[i][j]=j;//LRU栈，初始化时，为0,1,2; 表示替换的先后为0,1,2
                //end
				`endif
            end
            
        end
        for( integer k=0; k<LINE_SIZE; k++)
            mem_wr_line[k] <= 0;
        mem_wr_addr <= 0;
        {mem_rd_tag_addr, mem_rd_set_addr} <= 0;
        rd_data <= 0;
    end else begin
        case(cache_stat)
        IDLE:       begin
                        if( cache_hit ) begin
                            if(rd_req) begin    // 如果cache命中，并且是读请求，
                                rd_data <= cache_mem[set_addr][which_way][line_addr];   //则直接从cache中取出要读的数据
                            end else if(wr_req) begin // 如果cache命中，并且是写请求，
                                cache_mem[set_addr][which_way][line_addr] <= wr_data;   // 则直接向cache中写入数�?
                                dirty[set_addr][which_way] <= 1'b1;                     // 写数据的同时置脏�?
                            end 
                        end else begin
                            if(wr_req | rd_req) begin   // 如果 cache 未命中，并且有读写请求，则需要进行换�?
                                if( valid[set_addr][which_way] & dirty[set_addr][which_way] ) begin    // 如果 要换入的cache line 本来有效，且脏，则需要先将它换出
                                    cache_stat  <= SWAP_OUT;
                                    mem_wr_addr <= { cache_tags[set_addr][which_way], set_addr };
                                    mem_wr_line <= cache_mem[set_addr][which_way];
                                end else begin                                   // 反之，不�?要换出，直接换入
                                    cache_stat  <= SWAP_IN;
                                end
                                {mem_rd_tag_addr, mem_rd_set_addr} <= {tag_addr, set_addr};
                            end
                        end
                    end
        SWAP_OUT:   begin
                        if(mem_gnt) begin           // 如果主存握手信号有效，说明换出成功，跳到下一状�??
                            cache_stat <= SWAP_IN;
                        end
                    end
        SWAP_IN:    begin
                        if(mem_gnt) begin           // 如果主存握手信号有效，说明换入成功，跳到下一状�??
                            cache_stat <= SWAP_IN_OK;
                        end
                    end
        SWAP_IN_OK:begin           // 上一个周期换入成功，这周期将主存读出的line写入cache，并更新tag，置高valid，置低dirty
                        for( integer i=0; i<LINE_SIZE; i++)  cache_mem[mem_rd_set_addr][which_way][i] <= mem_rd_line[i];
                        cache_tags[mem_rd_set_addr][which_way] <= mem_rd_tag_addr;
                        valid     [mem_rd_set_addr][which_way] <= 1'b1;
                        dirty     [mem_rd_set_addr][which_way] <= 1'b0;
                        cache_stat <= IDLE;        // 回到就绪状�??
                   end
        endcase
    end
end

wire mem_rd_req = (cache_stat == SWAP_IN );
wire mem_wr_req = (cache_stat == SWAP_OUT);
wire [   MEM_ADDR_LEN-1 :0] mem_addr = mem_rd_req ? mem_rd_addr : ( mem_wr_req ? mem_wr_addr : 0);

assign miss = (rd_req | wr_req) & ~(cache_hit && cache_stat==IDLE) ;     // �? 有读写请求时，如果cache不处于就�?(IDLE)状�?�，或�?�未命中，则miss=1

main_mem #(     // slow main memory
    .LINE_ADDR_LEN  ( LINE_ADDR_LEN          ),
    .ADDR_LEN       ( MEM_ADDR_LEN           )
) main_mem_instance (
    .clk            ( clk                    ),
    .rst            ( rst                    ),
    .gnt            ( mem_gnt                ),
    .addr           ( mem_addr               ),
    .rd_req         ( mem_rd_req             ),
    .rd_line        ( mem_rd_line            ),
    .wr_req         ( mem_wr_req             ),
    .wr_line        ( mem_wr_line            )
);

endmodule





