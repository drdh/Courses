module top(clk,rst,sw,seg,an);
	input clk,rst;
	input [3:0]sw;
	output [7:0]seg;
	output [3:0]an;
	
	wire en;
	wire [3:0]addr_b,addr_c;
	wire [15:0]data_c,q_a,q_b;
	
	display U1(clk,q_a,seg,an);
	regfile U2(clk,rst,sw,addr_b,addr_c,data_c,en,q_a,q_b);
	inc U3(clk,q_b,addr_b,addr_c,data_c,en);
endmodule

module display(clk,data,seg,an);
	input clk;
	input [15:0]data;
	output reg [3:0]an;
	output reg [7:0]seg;
	
	reg [3:0]disp;
	
	reg [18:0]count=0;
	always@(posedge clk)			
		count<=count+1;	
		
	always@(posedge clk)
		case(count[18:17])
		2'b00:begin
			an=4'b1110;
			disp=data[3:0];
		end
		2'b01:begin
			an=4'b1101;
			disp=data[7:4];
		end
		2'b10:begin
			an=4'b1011;
			disp=data[11:8];
		end
		2'b11:begin
			an=4'b0111;
			disp=data[15:12];
		end
	endcase
	
	always @(disp)
	case(disp)
		0:seg=8'b11000000;
		1:seg=8'b11111001;
		2:seg=8'b10100100;
		3:seg=8'b10110000;
		4:seg=8'b10011001;
		5:seg=8'b10010010;
		6:seg=8'b10000010;
		7:seg=8'b11111000;
		8:seg=8'b10000000;
		9:seg=8'b10010000;
	  10:seg=8'b10001000;
	  11:seg=8'b10000011;
	  12:seg=8'b11000110;
	  13:seg=8'b10100001;
	  14:seg=8'b10000110;
	  15:seg=8'b10001110;
default:seg=8'b11000000;
	endcase
endmodule

module regfile(clk,rst,addr_a,addr_b,addr_c,data_c,en,q_a,q_b);
	input clk,rst,en;
	input [3:0]addr_a,addr_b,addr_c;
	input [15:0]data_c;
	output [15:0]q_a,q_b;
	
	reg [15:0] file [0:15];
	integer i;
	
	always@(posedge clk,posedge rst)
	begin
	if(rst)
	begin
		for(i=0;i<16;i=i+1)
		begin
			file[i][15:12]=i;
			file[i][11:8]=i;
			file[i][7:0]=0;
		end
	end
	else if(en)
		file[addr_c]=data_c;
	else	i=0;
	end
	assign q_a=file[addr_a];
	assign q_b=file[addr_b];
endmodule

module inc(clk,data_in,addr_b,addr_c,data,en);
	input clk;
	input [15:0]data_in;
	output reg[3:0] addr_b,addr_c;
	output reg en=0;
	output reg[15:0]data;
	reg [19:0]count=0;
	reg [3:0]change=0;
	always@(posedge clk)
	begin
		if(count==20'd781_250)
			count<=0;
		else
			count<=count+1;
	end	

	always@(negedge count[19])
	begin
		if(~en)
		begin
		en=1;
		data=data_in+16'h0001;
		end
		else
		begin
			en=0;
			change=change+4'b0001;
			addr_b=change;
			addr_c=change;
		end
	end
endmodule