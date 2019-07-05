`timescale 1ns / 1ps
module test_v;

	// Inputs
	reg clk;
	reg reset;
	reg [64:1] key;
	reg [64:1] din;
	reg flag;

	// Outputs
	wire [64:1] dout;
	wire ready;

	// Instantiate the Unit Under Test (UUT)
	ARS_des uut (
		.clk(clk), 
		.reset(reset), 
		.key(key), 
		.din(din), 
		.flag(flag), 
		.dout(dout), 
		.ready(ready)
	);

	initial begin
		// Initialize Inputs
		clk = 0;
		reset = 0;
		key = 64'h0123456789abcdef;
		din = 64'h82bc228322dce089;
		flag = 1;

		// Wait 100 ns for global reset to finish
		#10;
        reset = 1;
		// Add stimulus here
	end
     always #4 clk = ~clk; 
endmodule





module ARS_des(clk,reset,key,din,flag,dout,ready);
 input [64:1] din;
input [64:1] key;
input clk,reset;
input flag;
output [64:1] dout;
output ready;
reg [64:1] data1;
reg [64:1] key1;
wire [32:1] dout1,dout2;
wire [48:1] key2;
reg [4:1] round;
reg [4:1] state;
reg ready;
reg [64:1] dout;
parameter s0=4'h0,s1=4'h1,s2= 4'h2,s3=4'h3,s4= 4'h4,s5= 4'h5,s6= 4'h6,s7= 4'h7,s8= 4'h8,s9= 4'h9,s10= 4'ha,s11= 4'hb,s12= 4'hc;
function [64:1]	IP;
input	[64:1]	IPin;
//reg		[7:0]	s0_o,s1_o,s2_o,s3_o;
begin
IP=	 {IPin[58],IPin[50],IPin[42],IPin[34],IPin[26],IPin[18],IPin[10],IPin[2],

      IPin[60],IPin[52],IPin[44],IPin[36],IPin[28],IPin[20],IPin[12],IPin[4],

      IPin[62],IPin[54],IPin[46],IPin[38],IPin[30],IPin[22],IPin[14],IPin[6],

      IPin[64],IPin[56],IPin[48],IPin[40],IPin[32],IPin[24],IPin[16],IPin[8],

      IPin[57],IPin[49],IPin[41],IPin[33],IPin[25],IPin[17],IPin[9],IPin[1],

      IPin[59],IPin[51],IPin[43],IPin[35],IPin[27],IPin[19],IPin[11],IPin[3],

      IPin[61],IPin[53],IPin[45],IPin[37],IPin[29],IPin[21],IPin[13],IPin[5],

      IPin[63],IPin[55],IPin[47],IPin[39],IPin[31],IPin[23],IPin[15],IPin[7]};

end
endfunction
function [64:1] IVIP;
input    [64:1]  IP_1in;
begin
IVIP = {IP_1in[40],IP_1in[8],IP_1in[48],IP_1in[16],IP_1in[56],IP_1in[24],IP_1in[64],IP_1in[32],

        IP_1in[39],IP_1in[7],IP_1in[47],IP_1in[15],IP_1in[55],IP_1in[23],IP_1in[63],IP_1in[31],

        IP_1in[38],IP_1in[6],IP_1in[46],IP_1in[14],IP_1in[54],IP_1in[22],IP_1in[62],IP_1in[30],

        IP_1in[37],IP_1in[5],IP_1in[45],IP_1in[13],IP_1in[53],IP_1in[21],IP_1in[61],IP_1in[29],

        IP_1in[36],IP_1in[4],IP_1in[44],IP_1in[12],IP_1in[52],IP_1in[20],IP_1in[60],IP_1in[28],

        IP_1in[35],IP_1in[3],IP_1in[43],IP_1in[11],IP_1in[51],IP_1in[19],IP_1in[59],IP_1in[27],

        IP_1in[34],IP_1in[2],IP_1in[42],IP_1in[10],IP_1in[50],IP_1in[18],IP_1in[58],IP_1in[26],

        IP_1in[33],IP_1in[1],IP_1in[41],IP_1in[9],IP_1in[49],IP_1in[17],IP_1in[57],IP_1in[25]};
end
endfunction
always @(posedge clk) 
begin
if(reset == 0) begin  
             state	=	s0;
		     ready	=	0;
		     round=	(flag == 0)? 4'h0  : 4'hf;
             end 
else 
case(state)
		s0 : begin 
		     data1  = din;  
		     key1  = key; 
		     state = s1; 
		     end
		s1 : begin 
				data1 = IP({data1[64:33], data1[32:1]});
				if(flag == 0) 
				begin 
				round = 0; 
				state = s2; 
				end
				else          
				begin 
				round = 15;state = s4; 
				end
			 end
		s2 : begin //encription
				data1 = {dout1, dout2};
				round = round + 4'h1;
				state = s3;
			 end
		s3 : begin 
				if(round == 15) 
					begin 
						data1 = IVIP({dout2, dout1}); 
						ready = 1; 
						state = s6;
					end
				else 
				state = s2; 
			 end
		s4 : begin//decription
				data1 = {dout1, dout2};
				round = round - 4'h1;
				state = s5;
			 end
		s5 : begin 
				if(round == 0) 
					begin 
						data1 = IVIP({dout2, dout1}); 
						ready = 1; 
						state = s6;
					end
				else state = s4; 
			 end
		s6: begin 
				dout = data1; 
				state = s7; 
			 end
		s7: begin 
		      ready = 1; 
		      state = s7;
		      end
		default : state = s7;
		endcase
end
ARS_ENDEcrp 	c1 (.OUTR(dout2),  .OUTL(dout1),  .INR(data1[32:1]),  .INL(data1[64:33]),  .K_sub(key2));
ARS_keyExtension	c2(.K_sub(key2), .key(key1), .round(round));


endmodule






module ARS_ENDEcrp(OUTR, OUTL, INR, INL, K_sub);
output	[32:1]	OUTR, OUTL;
input	[32:1]	INR, INL;
input	[48:1]	K_sub;

wire	[48:1] E;
wire	[48:1] X;
wire	[48:1] S;

assign E[48:1] = {	INR[32], INR[1], INR[2], INR[3], INR[4], INR[5], INR[4], INR[5],
			INR[6], INR[7], INR[8], INR[9], INR[8], INR[9], INR[10], INR[11],
			INR[12], INR[13], INR[12], INR[13], INR[14], INR[15], INR[16],
			INR[17], INR[16], INR[17], INR[18], INR[19], INR[20], INR[21],
			INR[20], INR[21], INR[22], INR[23], INR[24], INR[25], INR[24],
			INR[25], INR[26], INR[27], INR[28], INR[29], INR[28], INR[29],
			INR[30], INR[31], INR[32], INR[1]};

assign X = E ^ K_sub;

ARS_sbox1 u0( .addr(X[06:01]), .dout(S[04:01]) );
ARS_sbox2 u1( .addr(X[12:07]), .dout(S[08:05]) );
ARS_sbox3 u2( .addr(X[18:13]), .dout(S[12:09]) );
ARS_sbox4 u3( .addr(X[24:19]), .dout(S[16:13]) );
ARS_sbox5 u4( .addr(X[30:25]), .dout(S[20:17]) );
ARS_sbox6 u5( .addr(X[36:31]), .dout(S[24:21]) );
ARS_sbox7 u6( .addr(X[42:37]), .dout(S[28:25]) );
ARS_sbox8 u7( .addr(X[48:43]), .dout(S[32:29]) );

assign OUTR = {	S[16], S[7], S[20], S[21], S[29], S[12], S[28],
			S[17], S[1], S[15], S[23], S[26], S[5], S[18],
			S[31], S[10], S[2], S[8], S[24], S[14], S[32],
			S[27], S[3], S[9], S[19], S[13], S[30], S[6],
			S[22], S[11], S[4], S[25]}^ INL;

assign OUTL = INR;


endmodule






module ARS_sbox1(addr, dout);
input	[6:1] addr;
output	[4:1] dout;
reg	[4:1] dout;

always @(addr) begin
    case ({addr[6],addr[1], addr[5:2]})	
 
         0:  dout =  14;
         1:  dout =   4;
         2:  dout =  13;
         3:  dout =   1;
         4:  dout =   2;
         5:  dout =  15;
         6:  dout =  11;
         7:  dout =   8;
         8:  dout =   3;
         9:  dout =  10;
        10:  dout =   6;
        11:  dout =  12;
        12:  dout =   5;
        13:  dout =   9;
        14:  dout =   0;
        15:  dout =   7;
        16:  dout =   0;
        17:  dout =  15;
        18:  dout =   7;
        19:  dout =   4;
        20:  dout =  14;
        21:  dout =   2;
        22:  dout =  13;
        23:  dout =   1;
        24:  dout =  10;
        25:  dout =   6;
        26:  dout =  12;
        27:  dout =  11;
        28:  dout =   9;
        29:  dout =   5;
        30:  dout =   3;
        31:  dout =   8;
        32:  dout =   4;
        33:  dout =   1;
        34:  dout =  14;
        35:  dout =   8;
        36:  dout =  13;
        37:  dout =   6;
        38:  dout =   2;
        39:  dout =  11;
        40:  dout =  15;
        41:  dout =  12;
        42:  dout =   9;
        43:  dout =   7;
        44:  dout =   3;
        45:  dout =  10;
        46:  dout =   5;
        47:  dout =   0;
        48:  dout =  15;
        49:  dout =  12;
        50:  dout =   8;
        51:  dout =   2;
        52:  dout =   4;
        53:  dout =   9;
        54:  dout =   1;
        55:  dout =   7;
        56:  dout =   5;
        57:  dout =  11;
        58:  dout =   3;
        59:  dout =  14;
        60:  dout =  10;
        61:  dout =   0;
        62:  dout =   6;
        63:  dout =  13;
 
    endcase
    end


endmodule





module ARS_sbox2(addr, dout);
input	[6:1] addr;
output	[4:1] dout;
reg	[4:1] dout;

always @(addr) begin
    case ({addr[6],addr[1], addr[5:2]})
 
         0:  dout = 15;
         1:  dout =  1;
         2:  dout =  8;
         3:  dout = 14;
         4:  dout =  6;
         5:  dout = 11;
         6:  dout =  3;
         7:  dout =  4;
         8:  dout =  9;
         9:  dout =  7;
        10:  dout =  2;
        11:  dout = 13;
        12:  dout = 12;
        13:  dout =  0;
        14:  dout =  5;
        15:  dout = 10;
        16:  dout =  3;
        17:  dout = 13;
        18:  dout =  4;
        19:  dout =  7;
        20:  dout = 15;
        21:  dout =  2;
        22:  dout =  8;
        23:  dout = 14;
        24:  dout = 12;
        25:  dout =  0;
        26:  dout =  1;
        27:  dout = 10;
        28:  dout =  6;
        29:  dout =  9;
        30:  dout = 11;
        31:  dout =  5;
        32:  dout =  0;
        33:  dout = 14;
        34:  dout =  7;
        35:  dout = 11;
        36:  dout = 10;
        37:  dout =  4;
        38:  dout = 13;
        39:  dout =  1;
        40:  dout =  5;
        41:  dout =  8;
        42:  dout = 12;
        43:  dout =  6;
        44:  dout =  9;
        45:  dout =  3;
        46:  dout =  2;
        47:  dout = 15;
        48:  dout = 13;
        49:  dout =  8;
        50:  dout = 10;
        51:  dout =  1;
        52:  dout =  3;
        53:  dout = 15;
        54:  dout =  4;
        55:  dout =  2;
        56:  dout = 11;
        57:  dout =  6;
        58:  dout =  7;
        59:  dout = 12;
        60:  dout =  0;
        61:  dout =  5;
        62:  dout = 14;
        63:  dout =  9;
 
    endcase
    end


endmodule




module ARS_sbox3(addr, dout);
input	[6:1] addr;
output	[4:1] dout;
reg	[4:1] dout;

always @(addr) begin
    case ({addr[6],addr[1], addr[5:2]})
 
         0:  dout = 10;
         1:  dout =  0;
         2:  dout =  9;
         3:  dout = 14;
         4:  dout =  6;
         5:  dout =  3;
         6:  dout = 15;
         7:  dout =  5;
         8:  dout =  1;
         9:  dout = 13;
        10:  dout = 12;
        11:  dout =  7;
        12:  dout = 11;
        13:  dout =  4;
        14:  dout =  2;
        15:  dout =  8;
        16:  dout = 13;
        17:  dout =  7;
        18:  dout =  0;
        19:  dout =  9;
        20:  dout =  3;
        21:  dout =  4;
        22:  dout =  6;
        23:  dout = 10;
        24:  dout =  2;
        25:  dout =  8;
        26:  dout =  5;
        27:  dout = 14;
        28:  dout = 12;
        29:  dout = 11;
        30:  dout = 15;
        31:  dout =  1;
        32:  dout = 13;
        33:  dout =  6;
        34:  dout =  4;
        35:  dout =  9;
        36:  dout =  8;
        37:  dout = 15;
        38:  dout =  3;
        39:  dout =  0;
        40:  dout = 11;
        41:  dout =  1;
        42:  dout =  2;
        43:  dout = 12;
        44:  dout =  5;
        45:  dout = 10;
        46:  dout = 14;
        47:  dout =  7;
        48:  dout =  1;
        49:  dout = 10;
        50:  dout = 13;
        51:  dout =  0;
        52:  dout =  6;
        53:  dout =  9;
        54:  dout =  8;
        55:  dout =  7;
        56:  dout =  4;
        57:  dout = 15;
        58:  dout = 14;
        59:  dout =  3;
        60:  dout = 11;
        61:  dout =  5;
        62:  dout =  2;
        63:  dout = 12;
 
    endcase
    end


endmodule






module ARS_sbox4(addr, dout);
input	[6:1] addr;
output	[4:1] dout;
reg	[4:1] dout;

always @(addr) begin
    case ({addr[6],addr[1], addr[5:2]})
 
         0:  dout =  7;
         1:  dout = 13;
         2:  dout = 14;
         3:  dout =  3;
         4:  dout =  0;
         5:  dout =  6;
         6:  dout =  9;
         7:  dout = 10;
         8:  dout =  1;
         9:  dout =  2;
        10:  dout =  8;
        11:  dout =  5;
        12:  dout = 11;
        13:  dout = 12;
        14:  dout =  4;
        15:  dout = 15;
        16:  dout = 13;
        17:  dout =  8;
        18:  dout = 11;
        19:  dout =  5;
        20:  dout =  6;
        21:  dout = 15;
        22:  dout =  0;
        23:  dout =  3;
        24:  dout =  4;
        25:  dout =  7;
        26:  dout =  2;
        27:  dout = 12;
        28:  dout =  1;
        29:  dout = 10;
        30:  dout = 14;
        31:  dout =  9;
        32:  dout = 10;
        33:  dout =  6;
        34:  dout =  9;
        35:  dout =  0;
        36:  dout = 12;
        37:  dout = 11;
        38:  dout =  7;
        39:  dout = 13;
        40:  dout = 15;
        41:  dout =  1;
        42:  dout =  3;
        43:  dout = 14;
        44:  dout =  5;
        45:  dout =  2;
        46:  dout =  8;
        47:  dout =  4;
        48:  dout =  3;
        49:  dout = 15;
        50:  dout =  0;
        51:  dout =  6;
        52:  dout = 10;
        53:  dout =  1;
        54:  dout = 13;
        55:  dout =  8;
        56:  dout =  9;
        57:  dout =  4;
        58:  dout =  5;
        59:  dout = 11;
        60:  dout = 12;
        61:  dout =  7;
        62:  dout =  2;
        63:  dout = 14;
 
    endcase
    end



endmodule





module ARS_sbox5(addr, dout);
input	[6:1] addr;
output	[4:1] dout;
reg	[4:1] dout;

always @(addr) begin
    case ({addr[6],addr[1], addr[5:2]})	
 
         0:  dout =  2;
         1:  dout = 12;
         2:  dout =  4;
         3:  dout =  1;
         4:  dout =  7;
         5:  dout = 10;
         6:  dout = 11;
         7:  dout =  6;
         8:  dout =  8;
         9:  dout =  5;
        10:  dout =  3;
        11:  dout = 15;
        12:  dout = 13;
        13:  dout =  0;
        14:  dout = 14;
        15:  dout =  9;
        16:  dout = 14;
        17:  dout = 11;
        18:  dout =  2;
        19:  dout = 12;
        20:  dout =  4;
        21:  dout =  7;
        22:  dout = 13;
        23:  dout =  1;
        24:  dout =  5;
        25:  dout =  0;
        26:  dout = 15;
        27:  dout = 10;
        28:  dout =  3;
        29:  dout =  9;
        30:  dout =  8;
        31:  dout =  6;
        32:  dout =  4;
        33:  dout =  2;
        34:  dout =  1;
        35:  dout = 11;
        36:  dout = 10;
        37:  dout = 13;
        38:  dout =  7;
        39:  dout =  8;
        40:  dout = 15;
        41:  dout =  9;
        42:  dout = 12;
        43:  dout =  5;
        44:  dout =  6;
        45:  dout =  3;
        46:  dout =  0;
        47:  dout = 14;
        48:  dout = 11;
        49:  dout =  8;
        50:  dout = 12;
        51:  dout =  7;
        52:  dout =  1;
        53:  dout = 14;
        54:  dout =  2;
        55:  dout = 13;
        56:  dout =  6;
        57:  dout = 15;
        58:  dout =  0;
        59:  dout =  9;
        60:  dout = 10;
        61:  dout =  4;
        62:  dout =  5;
        63:  dout =  3;
 
    endcase
    end


endmodule




module ARS_sbox6(addr, dout);
input	[6:1] addr;
output	[4:1] dout;
reg	[4:1] dout;

always @(addr) begin
    case ({addr[6],addr[1], addr[5:2]})	
 
         0:  dout = 12;
         1:  dout =  1;
         2:  dout = 10;
         3:  dout = 15;
         4:  dout =  9;
         5:  dout =  2;
         6:  dout =  6;
         7:  dout =  8;
         8:  dout =  0;
         9:  dout = 13;
        10:  dout =  3;
        11:  dout =  4;
        12:  dout = 14;
        13:  dout =  7;
        14:  dout =  5;
        15:  dout = 11;
        16:  dout = 10;
        17:  dout = 15;
        18:  dout =  4;
        19:  dout =  2;
        20:  dout =  7;
        21:  dout = 12;
        22:  dout =  9;
        23:  dout =  5;
        24:  dout =  6;
        25:  dout =  1;
        26:  dout = 13;
        27:  dout = 14;
        28:  dout =  0;
        29:  dout = 11;
        30:  dout =  3;
        31:  dout =  8;
        32:  dout =  9;
        33:  dout = 14;
        34:  dout = 15;
        35:  dout =  5;
        36:  dout =  2;
        37:  dout =  8;
        38:  dout = 12;
        39:  dout =  3;
        40:  dout =  7;
        41:  dout =  0;
        42:  dout =  4;
        43:  dout = 10;
        44:  dout =  1;
        45:  dout = 13;
        46:  dout = 11;
        47:  dout =  6;
        48:  dout =  4;
        49:  dout =  3;
        50:  dout =  2;
        51:  dout = 12;
        52:  dout =  9;
        53:  dout =  5;
        54:  dout = 15;
        55:  dout = 10;
        56:  dout = 11;
        57:  dout = 14;
        58:  dout =  1;
        59:  dout =  7;
        60:  dout =  6;
        61:  dout =  0;
        62:  dout =  8;
        63:  dout = 13;
 
    endcase
    end



endmodule



module ARS_sbox7(addr, dout);
input	[6:1] addr;
output	[4:1] dout;
reg	[4:1] dout;

always @(addr) begin
    case ({addr[6],addr[1], addr[5:2]})	
 
         0:  dout =  4;
         1:  dout = 11;
         2:  dout =  2;
         3:  dout = 14;
         4:  dout = 15;
         5:  dout =  0;
         6:  dout =  8;
         7:  dout = 13;
         8:  dout =  3;
         9:  dout = 12;
        10:  dout =  9;
        11:  dout =  7;
        12:  dout =  5;
        13:  dout = 10;
        14:  dout =  6;
        15:  dout =  1;
        16:  dout = 13;
        17:  dout =  0;
        18:  dout = 11;
        19:  dout =  7;
        20:  dout =  4;
        21:  dout =  9;
        22:  dout =  1;
        23:  dout = 10;
        24:  dout = 14;
        25:  dout =  3;
        26:  dout =  5;
        27:  dout = 12;
        28:  dout =  2;
        29:  dout = 15;
        30:  dout =  8;
        31:  dout =  6;
        32:  dout =  1;
        33:  dout =  4;
        34:  dout = 11;
        35:  dout = 13;
        36:  dout = 12;
        37:  dout =  3;
        38:  dout =  7;
        39:  dout = 14;
        40:  dout = 10;
        41:  dout = 15;
        42:  dout =  6;
        43:  dout =  8;
        44:  dout =  0;
        45:  dout =  5;
        46:  dout =  9;
        47:  dout =  2;
        48:  dout =  6;
        49:  dout = 11;
        50:  dout = 13;
        51:  dout =  8;
        52:  dout =  1;
        53:  dout =  4;
        54:  dout = 10;
        55:  dout =  7;
        56:  dout =  9;
        57:  dout =  5;
        58:  dout =  0;
        59:  dout = 15;
        60:  dout = 14;
        61:  dout =  2;
        62:  dout =  3;
        63:  dout = 12;
 

    endcase
    end


endmodule



module ARS_sbox8(addr, dout);
input	[6:1] addr;
output	[4:1] dout;
reg	[4:1] dout;

always @(addr) begin
    case ({addr[6],addr[1], addr[5:2]})	
 
         0:  dout = 13;
         1:  dout =  2;
         2:  dout =  8;
         3:  dout =  4;
         4:  dout =  6;
         5:  dout = 15;
         6:  dout = 11;
         7:  dout =  1;
         8:  dout = 10;
         9:  dout =  9;
        10:  dout =  3;
        11:  dout = 14;
        12:  dout =  5;
        13:  dout =  0;
        14:  dout = 12;
        15:  dout =  7;
        16:  dout =  1;
        17:  dout = 15;
        18:  dout = 13;
        19:  dout =  8;
        20:  dout = 10;
        21:  dout =  3;
        22:  dout =  7;
        23:  dout =  4;
        24:  dout = 12;
        25:  dout =  5;
        26:  dout =  6;
        27:  dout = 11;
        28:  dout =  0;
        29:  dout = 14;
        30:  dout =  9;
        31:  dout =  2;
        32:  dout =  7;
        33:  dout = 11;
        34:  dout =  4;
        35:  dout =  1;
        36:  dout =  9;
        37:  dout = 12;
        38:  dout = 14;
        39:  dout =  2;
        40:  dout =  0;
        41:  dout =  6;
        42:  dout = 10;
        43:  dout = 13;
        44:  dout = 15;
        45:  dout =  3;
        46:  dout =  5;
        47:  dout =  8;
        48:  dout =  2;
        49:  dout =  1;
        50:  dout = 14;
        51:  dout =  7;
        52:  dout =  4;
        53:  dout = 10;
        54:  dout =  8;
        55:  dout = 13;
        56:  dout = 15;
        57:  dout = 12;
        58:  dout =  9;
        59:  dout =  0;
        60:  dout =  3;
        61:  dout =  5;
        62:  dout =  6;
        63:  dout = 11;
 
    endcase
    end


endmodule




module ARS_keyExtension(K_sub, key, round);
output	[1:48]	K_sub;
input	[1:64]	key;
input	[3:0]	round;

wire [55:0]	K;
reg  [1:48]	K_sub;
wire [1:48]	K1, K2, K3, K4, K5, K6, K7, K8, K9;
wire [1:48]	K10, K11, K12, K13, K14, K15, K16;

assign K = {		key[01 : 07], key[09:15], key[17:23], key[25:31],
				key[33: 39], key[41:47], key[49:55], key[57:63]
		   }; 

always @(K1 or K2 or K3 or K4 or K5 or K6 or K7 or K8 or K9 or K10
              or K11 or K12 or K13 or K14 or K15 or K16 or round)
	case (round)		// synopsys full_case parallel_case
            0:  K_sub = K1;
            1:  K_sub = K2;
            2:  K_sub = K3;
            3:  K_sub = K4;
            4:  K_sub = K5;
            5:  K_sub = K6;
            6:  K_sub = K7;
            7:  K_sub = K8;
            8:  K_sub = K9;
            9:  K_sub = K10;
            10: K_sub = K11;
            11: K_sub = K12;
            12: K_sub = K13;
            13: K_sub = K14;
            14: K_sub = K15;
            15: K_sub = K16;
	endcase

 
assign K1[1] = K[47];
assign K1[2] = K[11];
assign K1[3] = K[26];
assign K1[4] = K[3];
assign K1[5] = K[13];
assign K1[6] = K[41];
assign K1[7] = K[27];
assign K1[8] = K[6];
assign K1[9] = K[54];
assign K1[10] = K[48];
assign K1[11] = K[39];
assign K1[12] = K[19];
assign K1[13] = K[53];
assign K1[14] = K[25];
assign K1[15] = K[33];
assign K1[16] = K[34];
assign K1[17] = K[17];
assign K1[18] = K[5];
assign K1[19] = K[4];
assign K1[20] = K[55];
assign K1[21] = K[24];
assign K1[22] = K[32];
assign K1[23] = K[40];
assign K1[24] = K[20];
assign K1[25] = K[36];
assign K1[26] = K[31];
assign K1[27] = K[21];
assign K1[28] = K[8];
assign K1[29] = K[23];
assign K1[30] = K[52];
assign K1[31] = K[14];
assign K1[32] = K[29];
assign K1[33] = K[51];
assign K1[34] = K[9];
assign K1[35] = K[35];
assign K1[36] = K[30];
assign K1[37] = K[2];
assign K1[38] = K[37];
assign K1[39] = K[22];
assign K1[40] = K[0];
assign K1[41] = K[42];
assign K1[42] = K[38];
assign K1[43] = K[16];
assign K1[44] = K[43];
assign K1[45] = K[44];
assign K1[46] = K[1];
assign K1[47] = K[7];
assign K1[48] = K[28];
assign K2[1] = K[54];
assign K2[2] = K[18];
assign K2[3] = K[33];
assign K2[4] = K[10];
assign K2[5] = K[20];
assign K2[6] = K[48];
assign K2[7] = K[34];
assign K2[8] = K[13];
assign K2[9] = K[4];
assign K2[10] = K[55];
assign K2[11] = K[46];
assign K2[12] = K[26];
assign K2[13] = K[3];
assign K2[14] = K[32];
assign K2[15] = K[40];
assign K2[16] = K[41];
assign K2[17] = K[24];
assign K2[18] = K[12];
assign K2[19] = K[11];
assign K2[20] = K[5];
assign K2[21] = K[6];
assign K2[22] = K[39];
assign K2[23] = K[47];
assign K2[24] = K[27];
assign K2[25] = K[43];
assign K2[26] = K[38];
assign K2[27] = K[28];
assign K2[28] = K[15];
assign K2[29] = K[30];
assign K2[30] = K[0];
assign K2[31] = K[21];
assign K2[32] = K[36];
assign K2[33] = K[31];
assign K2[34] = K[16];
assign K2[35] = K[42];
assign K2[36] = K[37];
assign K2[37] = K[9];
assign K2[38] = K[44];
assign K2[39] = K[29];
assign K2[40] = K[7];
assign K2[41] = K[49];
assign K2[42] = K[45];
assign K2[43] = K[23];
assign K2[44] = K[50];
assign K2[45] = K[51];
assign K2[46] = K[8];
assign K2[47] = K[14];
assign K2[48] = K[35];
assign K3[1] = K[11];
assign K3[2] = K[32];
assign K3[3] = K[47];
assign K3[4] = K[24];
assign K3[5] = K[34];
assign K3[6] = K[5];
assign K3[7] = K[48];
assign K3[8] = K[27];
assign K3[9] = K[18];
assign K3[10] = K[12];
assign K3[11] = K[3];
assign K3[12] = K[40];
assign K3[13] = K[17];
assign K3[14] = K[46];
assign K3[15] = K[54];
assign K3[16] = K[55];
assign K3[17] = K[13];
assign K3[18] = K[26];
assign K3[19] = K[25];
assign K3[20] = K[19];
assign K3[21] = K[20];
assign K3[22] = K[53];
assign K3[23] = K[4];
assign K3[24] = K[41];
assign K3[25] = K[2];
assign K3[26] = K[52];
assign K3[27] = K[42];
assign K3[28] = K[29];
assign K3[29] = K[44];
assign K3[30] = K[14];
assign K3[31] = K[35];
assign K3[32] = K[50];
assign K3[33] = K[45];
assign K3[34] = K[30];
assign K3[35] = K[1];
assign K3[36] = K[51];
assign K3[37] = K[23];
assign K3[38] = K[31];
assign K3[39] = K[43];
assign K3[40] = K[21];
assign K3[41] = K[8];
assign K3[42] = K[0];
assign K3[43] = K[37];
assign K3[44] = K[9];
assign K3[45] = K[38];
assign K3[46] = K[22];
assign K3[47] = K[28];
assign K3[48] = K[49];
assign K4[1] = K[25];
assign K4[2] = K[46];
assign K4[3] = K[4];
assign K4[4] = K[13];
assign K4[5] = K[48];
assign K4[6] = K[19];
assign K4[7] = K[5];
assign K4[8] = K[41];
assign K4[9] = K[32];
assign K4[10] = K[26];
assign K4[11] = K[17];
assign K4[12] = K[54];
assign K4[13] = K[6];
assign K4[14] = K[3];
assign K4[15] = K[11];
assign K4[16] = K[12];
assign K4[17] = K[27];
assign K4[18] = K[40];
assign K4[19] = K[39];
assign K4[20] = K[33];
assign K4[21] = K[34];
assign K4[22] = K[10];
assign K4[23] = K[18];
assign K4[24] = K[55];
assign K4[25] = K[16];
assign K4[26] = K[7];
assign K4[27] = K[1];
assign K4[28] = K[43];
assign K4[29] = K[31];
assign K4[30] = K[28];
assign K4[31] = K[49];
assign K4[32] = K[9];
assign K4[33] = K[0];
assign K4[34] = K[44];
assign K4[35] = K[15];
assign K4[36] = K[38];
assign K4[37] = K[37];
assign K4[38] = K[45];
assign K4[39] = K[2];
assign K4[40] = K[35];
assign K4[41] = K[22];
assign K4[42] = K[14];
assign K4[43] = K[51];
assign K4[44] = K[23];
assign K4[45] = K[52];
assign K4[46] = K[36];
assign K4[47] = K[42];
assign K4[48] = K[8];
assign K5[1] = K[39];
assign K5[2] = K[3];
assign K5[3] = K[18];
assign K5[4] = K[27];
assign K5[5] = K[5];
assign K5[6] = K[33];
assign K5[7] = K[19];
assign K5[8] = K[55];
assign K5[9] = K[46];
assign K5[10] = K[40];
assign K5[11] = K[6];
assign K5[12] = K[11];
assign K5[13] = K[20];
assign K5[14] = K[17];
assign K5[15] = K[25];
assign K5[16] = K[26];
assign K5[17] = K[41];
assign K5[18] = K[54];
assign K5[19] = K[53];
assign K5[20] = K[47];
assign K5[21] = K[48];
assign K5[22] = K[24];
assign K5[23] = K[32];
assign K5[24] = K[12];
assign K5[25] = K[30];
assign K5[26] = K[21];
assign K5[27] = K[15];
assign K5[28] = K[2];
assign K5[29] = K[45];
assign K5[30] = K[42];
assign K5[31] = K[8];
assign K5[32] = K[23];
assign K5[33] = K[14];
assign K5[34] = K[31];
assign K5[35] = K[29];
assign K5[36] = K[52];
assign K5[37] = K[51];
assign K5[38] = K[0];
assign K5[39] = K[16];
assign K5[40] = K[49];
assign K5[41] = K[36];
assign K5[42] = K[28];
assign K5[43] = K[38];
assign K5[44] = K[37];
assign K5[45] = K[7];
assign K5[46] = K[50];
assign K5[47] = K[1];
assign K5[48] = K[22];
assign K6[1] = K[53];
assign K6[2] = K[17];
assign K6[3] = K[32];
assign K6[4] = K[41];
assign K6[5] = K[19];
assign K6[6] = K[47];
assign K6[7] = K[33];
assign K6[8] = K[12];
assign K6[9] = K[3];
assign K6[10] = K[54];
assign K6[11] = K[20];
assign K6[12] = K[25];
assign K6[13] = K[34];
assign K6[14] = K[6];
assign K6[15] = K[39];
assign K6[16] = K[40];
assign K6[17] = K[55];
assign K6[18] = K[11];
assign K6[19] = K[10];
assign K6[20] = K[4];
assign K6[21] = K[5];
assign K6[22] = K[13];
assign K6[23] = K[46];
assign K6[24] = K[26];
assign K6[25] = K[44];
assign K6[26] = K[35];
assign K6[27] = K[29];
assign K6[28] = K[16];
assign K6[29] = K[0];
assign K6[30] = K[1];
assign K6[31] = K[22];
assign K6[32] = K[37];
assign K6[33] = K[28];
assign K6[34] = K[45];
assign K6[35] = K[43];
assign K6[36] = K[7];
assign K6[37] = K[38];
assign K6[38] = K[14];
assign K6[39] = K[30];
assign K6[40] = K[8];
assign K6[41] = K[50];
assign K6[42] = K[42];
assign K6[43] = K[52];
assign K6[44] = K[51];
assign K6[45] = K[21];
assign K6[46] = K[9];
assign K6[47] = K[15];
assign K6[48] = K[36];
assign K7[1] = K[10];
assign K7[2] = K[6];
assign K7[3] = K[46];
assign K7[4] = K[55];
assign K7[5] = K[33];
assign K7[6] = K[4];
assign K7[7] = K[47];
assign K7[8] = K[26];
assign K7[9] = K[17];
assign K7[10] = K[11];
assign K7[11] = K[34];
assign K7[12] = K[39];
assign K7[13] = K[48];
assign K7[14] = K[20];
assign K7[15] = K[53];
assign K7[16] = K[54];
assign K7[17] = K[12];
assign K7[18] = K[25];
assign K7[19] = K[24];
assign K7[20] = K[18];
assign K7[21] = K[19];
assign K7[22] = K[27];
assign K7[23] = K[3];
assign K7[24] = K[40];
assign K7[25] = K[31];
assign K7[26] = K[49];
assign K7[27] = K[43];
assign K7[28] = K[30];
assign K7[29] = K[14];
assign K7[30] = K[15];
assign K7[31] = K[36];
assign K7[32] = K[51];
assign K7[33] = K[42];
assign K7[34] = K[0];
assign K7[35] = K[2];
assign K7[36] = K[21];
assign K7[37] = K[52];
assign K7[38] = K[28];
assign K7[39] = K[44];
assign K7[40] = K[22];
assign K7[41] = K[9];
assign K7[42] = K[1];
assign K7[43] = K[7];
assign K7[44] = K[38];
assign K7[45] = K[35];
assign K7[46] = K[23];
assign K7[47] = K[29];
assign K7[48] = K[50];
assign K8[1] = K[24];
assign K8[2] = K[20];
assign K8[3] = K[3];
assign K8[4] = K[12];
assign K8[5] = K[47];
assign K8[6] = K[18];
assign K8[7] = K[4];
assign K8[8] = K[40];
assign K8[9] = K[6];
assign K8[10] = K[25];
assign K8[11] = K[48];
assign K8[12] = K[53];
assign K8[13] = K[5];
assign K8[14] = K[34];
assign K8[15] = K[10];
assign K8[16] = K[11];
assign K8[17] = K[26];
assign K8[18] = K[39];
assign K8[19] = K[13];
assign K8[20] = K[32];
assign K8[21] = K[33];
assign K8[22] = K[41];
assign K8[23] = K[17];
assign K8[24] = K[54];
assign K8[25] = K[45];
assign K8[26] = K[8];
assign K8[27] = K[2];
assign K8[28] = K[44];
assign K8[29] = K[28];
assign K8[30] = K[29];
assign K8[31] = K[50];
assign K8[32] = K[38];
assign K8[33] = K[1];
assign K8[34] = K[14];
assign K8[35] = K[16];
assign K8[36] = K[35];
assign K8[37] = K[7];
assign K8[38] = K[42];
assign K8[39] = K[31];
assign K8[40] = K[36];
assign K8[41] = K[23];
assign K8[42] = K[15];
assign K8[43] = K[21];
assign K8[44] = K[52];
assign K8[45] = K[49];
assign K8[46] = K[37];
assign K8[47] = K[43];
assign K8[48] = K[9];
assign K9[1] = K[6];
assign K9[2] = K[27];
assign K9[3] = K[10];
assign K9[4] = K[19];
assign K9[5] = K[54];
assign K9[6] = K[25];
assign K9[7] = K[11];
assign K9[8] = K[47];
assign K9[9] = K[13];
assign K9[10] = K[32];
assign K9[11] = K[55];
assign K9[12] = K[3];
assign K9[13] = K[12];
assign K9[14] = K[41];
assign K9[15] = K[17];
assign K9[16] = K[18];
assign K9[17] = K[33];
assign K9[18] = K[46];
assign K9[19] = K[20];
assign K9[20] = K[39];
assign K9[21] = K[40];
assign K9[22] = K[48];
assign K9[23] = K[24];
assign K9[24] = K[4];
assign K9[25] = K[52];
assign K9[26] = K[15];
assign K9[27] = K[9];
assign K9[28] = K[51];
assign K9[29] = K[35];
assign K9[30] = K[36];
assign K9[31] = K[2];
assign K9[32] = K[45];
assign K9[33] = K[8];
assign K9[34] = K[21];
assign K9[35] = K[23];
assign K9[36] = K[42];
assign K9[37] = K[14];
assign K9[38] = K[49];
assign K9[39] = K[38];
assign K9[40] = K[43];
assign K9[41] = K[30];
assign K9[42] = K[22];
assign K9[43] = K[28];
assign K9[44] = K[0];
assign K9[45] = K[1];
assign K9[46] = K[44];
assign K9[47] = K[50];
assign K9[48] = K[16];
assign K10[1] = K[20];
assign K10[2] = K[41];
assign K10[3] = K[24];
assign K10[4] = K[33];
assign K10[5] = K[11];
assign K10[6] = K[39];
assign K10[7] = K[25];
assign K10[8] = K[4];
assign K10[9] = K[27];
assign K10[10] = K[46];
assign K10[11] = K[12];
assign K10[12] = K[17];
assign K10[13] = K[26];
assign K10[14] = K[55];
assign K10[15] = K[6];
assign K10[16] = K[32];
assign K10[17] = K[47];
assign K10[18] = K[3];
assign K10[19] = K[34];
assign K10[20] = K[53];
assign K10[21] = K[54];
assign K10[22] = K[5];
assign K10[23] = K[13];
assign K10[24] = K[18];
assign K10[25] = K[7];
assign K10[26] = K[29];
assign K10[27] = K[23];
assign K10[28] = K[38];
assign K10[29] = K[49];
assign K10[30] = K[50];
assign K10[31] = K[16];
assign K10[32] = K[0];
assign K10[33] = K[22];
assign K10[34] = K[35];
assign K10[35] = K[37];
assign K10[36] = K[1];
assign K10[37] = K[28];
assign K10[38] = K[8];
assign K10[39] = K[52];
assign K10[40] = K[2];
assign K10[41] = K[44];
assign K10[42] = K[36];
assign K10[43] = K[42];
assign K10[44] = K[14];
assign K10[45] = K[15];
assign K10[46] = K[31];
assign K10[47] = K[9];
assign K10[48] = K[30];
assign K11[1] = K[34];
assign K11[2] = K[55];
assign K11[3] = K[13];
assign K11[4] = K[47];
assign K11[5] = K[25];
assign K11[6] = K[53];
assign K11[7] = K[39];
assign K11[8] = K[18];
assign K11[9] = K[41];
assign K11[10] = K[3];
assign K11[11] = K[26];
assign K11[12] = K[6];
assign K11[13] = K[40];
assign K11[14] = K[12];
assign K11[15] = K[20];
assign K11[16] = K[46];
assign K11[17] = K[4];
assign K11[18] = K[17];
assign K11[19] = K[48];
assign K11[20] = K[10];
assign K11[21] = K[11];
assign K11[22] = K[19];
assign K11[23] = K[27];
assign K11[24] = K[32];
assign K11[25] = K[21];
assign K11[26] = K[43];
assign K11[27] = K[37];
assign K11[28] = K[52];
assign K11[29] = K[8];
assign K11[30] = K[9];
assign K11[31] = K[30];
assign K11[32] = K[14];
assign K11[33] = K[36];
assign K11[34] = K[49];
assign K11[35] = K[51];
assign K11[36] = K[15];
assign K11[37] = K[42];
assign K11[38] = K[22];
assign K11[39] = K[7];
assign K11[40] = K[16];
assign K11[41] = K[31];
assign K11[42] = K[50];
assign K11[43] = K[1];
assign K11[44] = K[28];
assign K11[45] = K[29];
assign K11[46] = K[45];
assign K11[47] = K[23];
assign K11[48] = K[44];
assign K12[1] = K[48];
assign K12[2] = K[12];
assign K12[3] = K[27];
assign K12[4] = K[4];
assign K12[5] = K[39];
assign K12[6] = K[10];
assign K12[7] = K[53];
assign K12[8] = K[32];
assign K12[9] = K[55];
assign K12[10] = K[17];
assign K12[11] = K[40];
assign K12[12] = K[20];
assign K12[13] = K[54];
assign K12[14] = K[26];
assign K12[15] = K[34];
assign K12[16] = K[3];
assign K12[17] = K[18];
assign K12[18] = K[6];
assign K12[19] = K[5];
assign K12[20] = K[24];
assign K12[21] = K[25];
assign K12[22] = K[33];
assign K12[23] = K[41];
assign K12[24] = K[46];
assign K12[25] = K[35];
assign K12[26] = K[2];
assign K12[27] = K[51];
assign K12[28] = K[7];
assign K12[29] = K[22];
assign K12[30] = K[23];
assign K12[31] = K[44];
assign K12[32] = K[28];
assign K12[33] = K[50];
assign K12[34] = K[8];
assign K12[35] = K[38];
assign K12[36] = K[29];
assign K12[37] = K[1];
assign K12[38] = K[36];
assign K12[39] = K[21];
assign K12[40] = K[30];
assign K12[41] = K[45];
assign K12[42] = K[9];
assign K12[43] = K[15];
assign K12[44] = K[42];
assign K12[45] = K[43];
assign K12[46] = K[0];
assign K12[47] = K[37];
assign K12[48] = K[31];
assign K13[1] = K[5];
assign K13[2] = K[26];
assign K13[3] = K[41];
assign K13[4] = K[18];
assign K13[5] = K[53];
assign K13[6] = K[24];
assign K13[7] = K[10];
assign K13[8] = K[46];
assign K13[9] = K[12];
assign K13[10] = K[6];
assign K13[11] = K[54];
assign K13[12] = K[34];
assign K13[13] = K[11];
assign K13[14] = K[40];
assign K13[15] = K[48];
assign K13[16] = K[17];
assign K13[17] = K[32];
assign K13[18] = K[20];
assign K13[19] = K[19];
assign K13[20] = K[13];
assign K13[21] = K[39];
assign K13[22] = K[47];
assign K13[23] = K[55];
assign K13[24] = K[3];
assign K13[25] = K[49];
assign K13[26] = K[16];
assign K13[27] = K[38];
assign K13[28] = K[21];
assign K13[29] = K[36];
assign K13[30] = K[37];
assign K13[31] = K[31];
assign K13[32] = K[42];
assign K13[33] = K[9];
assign K13[34] = K[22];
assign K13[35] = K[52];
assign K13[36] = K[43];
assign K13[37] = K[15];
assign K13[38] = K[50];
assign K13[39] = K[35];
assign K13[40] = K[44];
assign K13[41] = K[0];
assign K13[42] = K[23];
assign K13[43] = K[29];
assign K13[44] = K[1];
assign K13[45] = K[2];
assign K13[46] = K[14];
assign K13[47] = K[51];
assign K13[48] = K[45];
assign K14[1] = K[19];
assign K14[2] = K[40];
assign K14[3] = K[55];
assign K14[4] = K[32];
assign K14[5] = K[10];
assign K14[6] = K[13];
assign K14[7] = K[24];
assign K14[8] = K[3];
assign K14[9] = K[26];
assign K14[10] = K[20];
assign K14[11] = K[11];
assign K14[12] = K[48];
assign K14[13] = K[25];
assign K14[14] = K[54];
assign K14[15] = K[5];
assign K14[16] = K[6];
assign K14[17] = K[46];
assign K14[18] = K[34];
assign K14[19] = K[33];
assign K14[20] = K[27];
assign K14[21] = K[53];
assign K14[22] = K[4];
assign K14[23] = K[12];
assign K14[24] = K[17];
assign K14[25] = K[8];
assign K14[26] = K[30];
assign K14[27] = K[52];
assign K14[28] = K[35];
assign K14[29] = K[50];
assign K14[30] = K[51];
assign K14[31] = K[45];
assign K14[32] = K[1];
assign K14[33] = K[23];
assign K14[34] = K[36];
assign K14[35] = K[7];
assign K14[36] = K[2];
assign K14[37] = K[29];
assign K14[38] = K[9];
assign K14[39] = K[49];
assign K14[40] = K[31];
assign K14[41] = K[14];
assign K14[42] = K[37];
assign K14[43] = K[43];
assign K14[44] = K[15];
assign K14[45] = K[16];
assign K14[46] = K[28];
assign K14[47] = K[38];
assign K14[48] = K[0];
assign K15[1] = K[33];
assign K15[2] = K[54];
assign K15[3] = K[12];
assign K15[4] = K[46];
assign K15[5] = K[24];
assign K15[6] = K[27];
assign K15[7] = K[13];
assign K15[8] = K[17];
assign K15[9] = K[40];
assign K15[10] = K[34];
assign K15[11] = K[25];
assign K15[12] = K[5];
assign K15[13] = K[39];
assign K15[14] = K[11];
assign K15[15] = K[19];
assign K15[16] = K[20];
assign K15[17] = K[3];
assign K15[18] = K[48];
assign K15[19] = K[47];
assign K15[20] = K[41];
assign K15[21] = K[10];
assign K15[22] = K[18];
assign K15[23] = K[26];
assign K15[24] = K[6];
assign K15[25] = K[22];
assign K15[26] = K[44];
assign K15[27] = K[7];
assign K15[28] = K[49];
assign K15[29] = K[9];
assign K15[30] = K[38];
assign K15[31] = K[0];
assign K15[32] = K[15];
assign K15[33] = K[37];
assign K15[34] = K[50];
assign K15[35] = K[21];
assign K15[36] = K[16];
assign K15[37] = K[43];
assign K15[38] = K[23];
assign K15[39] = K[8];
assign K15[40] = K[45];
assign K15[41] = K[28];
assign K15[42] = K[51];
assign K15[43] = K[2];
assign K15[44] = K[29];
assign K15[45] = K[30];
assign K15[46] = K[42];
assign K15[47] = K[52];
assign K15[48] = K[14];
assign K16[1] = K[40];
assign K16[2] = K[4];
assign K16[3] = K[19];
assign K16[4] = K[53];
assign K16[5] = K[6];
assign K16[6] = K[34];
assign K16[7] = K[20];
assign K16[8] = K[24];
assign K16[9] = K[47];
assign K16[10] = K[41];
assign K16[11] = K[32];
assign K16[12] = K[12];
assign K16[13] = K[46];
assign K16[14] = K[18];
assign K16[15] = K[26];
assign K16[16] = K[27];
assign K16[17] = K[10];
assign K16[18] = K[55];
assign K16[19] = K[54];
assign K16[20] = K[48];
assign K16[21] = K[17];
assign K16[22] = K[25];
assign K16[23] = K[33];
assign K16[24] = K[13];
assign K16[25] = K[29];
assign K16[26] = K[51];
assign K16[27] = K[14];
assign K16[28] = K[1];
assign K16[29] = K[16];
assign K16[30] = K[45];
assign K16[31] = K[7];
assign K16[32] = K[22];
assign K16[33] = K[44];
assign K16[34] = K[2];
assign K16[35] = K[28];
assign K16[36] = K[23];
assign K16[37] = K[50];
assign K16[38] = K[30];
assign K16[39] = K[15];
assign K16[40] = K[52];
assign K16[41] = K[35];
assign K16[42] = K[31];
assign K16[43] = K[9];
assign K16[44] = K[36];
assign K16[45] = K[37];
assign K16[46] = K[49];
assign K16[47] = K[0];
assign K16[48] = K[21];


endmodule