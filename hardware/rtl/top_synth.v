module line_buffer (
	clk,
	rst,
	valid_in,
	pixel_in,
	win_00,
	win_01,
	win_02,
	win_10,
	win_11,
	win_12,
	win_20,
	win_21,
	win_22,
	valid_out
);
	parameter DATA_WIDTH = 16;
	parameter TILE_WIDTH = 32;
	input wire clk;
	input wire rst;
	input wire valid_in;
	input wire [DATA_WIDTH - 1:0] pixel_in;
	output wire [DATA_WIDTH - 1:0] win_00;
	output wire [DATA_WIDTH - 1:0] win_01;
	output wire [DATA_WIDTH - 1:0] win_02;
	output wire [DATA_WIDTH - 1:0] win_10;
	output wire [DATA_WIDTH - 1:0] win_11;
	output wire [DATA_WIDTH - 1:0] win_12;
	output wire [DATA_WIDTH - 1:0] win_20;
	output wire [DATA_WIDTH - 1:0] win_21;
	output wire [DATA_WIDTH - 1:0] win_22;
	output reg valid_out;
	reg [DATA_WIDTH - 1:0] line_ram_0 [0:TILE_WIDTH - 1];
	reg [DATA_WIDTH - 1:0] line_ram_1 [0:TILE_WIDTH - 1];
	reg [$clog2(TILE_WIDTH) - 1:0] ptr;
	localparam delay = (2 * TILE_WIDTH) + 2;
	reg [$clog2(delay + 1) - 1:0] pixel_count;
	reg [DATA_WIDTH - 1:0] win [0:2][0:2];
	assign win_00 = win[0][0];
	assign win_01 = win[0][1];
	assign win_02 = win[0][2];
	assign win_10 = win[1][0];
	assign win_11 = win[1][1];
	assign win_12 = win[1][2];
	assign win_20 = win[2][0];
	assign win_21 = win[2][1];
	assign win_22 = win[2][2];
	always @(posedge clk)
		if (rst) begin
			ptr <= 1'sb0;
			pixel_count <= 1'sb0;
			valid_out <= 1'b0;
			win[0][0] <= 1'sb0;
			win[0][1] <= 1'sb0;
			win[0][2] <= 1'sb0;
			win[1][0] <= 1'sb0;
			win[1][1] <= 1'sb0;
			win[1][2] <= 1'sb0;
			win[2][0] <= 1'sb0;
			win[2][1] <= 1'sb0;
			win[2][2] <= 1'sb0;
		end
		else if (valid_in) begin
			win[0][0] <= win[0][1];
			win[0][1] <= win[0][2];
			win[1][0] <= win[1][1];
			win[1][1] <= win[1][2];
			win[2][0] <= win[2][1];
			win[2][1] <= win[2][2];
			win[0][2] <= line_ram_1[ptr];
			win[1][2] <= line_ram_0[ptr];
			win[2][2] <= pixel_in;
			line_ram_1[ptr] <= line_ram_0[ptr];
			line_ram_0[ptr] <= pixel_in;
			ptr <= (ptr == (TILE_WIDTH - 1) ? {$clog2(TILE_WIDTH) {1'sb0}} : ptr + 1'b1);
			if (pixel_count < delay) begin
				pixel_count <= pixel_count + 1'b1;
				valid_out <= 1'b0;
			end
			else
				valid_out <= 1'b1;
		end
		else
			valid_out <= 1'b0;
endmodule
module line_buffer_top (
	clk,
	rst,
	s_axis_tvalid,
	s_axis_tdata,
	s_axis_trdy,
	m_axis_trdy,
	m_axis_tvalid,
	win_00,
	win_01,
	win_02,
	win_10,
	win_11,
	win_12,
	win_20,
	win_21,
	win_22
);
	parameter DATA_WIDTH = 16;
	parameter TILE_WIDTH = 32;
	parameter TILE_HEIGHT = 32;
	input wire clk;
	input wire rst;
	input wire s_axis_tvalid;
	input wire [DATA_WIDTH - 1:0] s_axis_tdata;
	output wire s_axis_trdy;
	input wire m_axis_trdy;
	output wire m_axis_tvalid;
	output wire [DATA_WIDTH - 1:0] win_00;
	output wire [DATA_WIDTH - 1:0] win_01;
	output wire [DATA_WIDTH - 1:0] win_02;
	output wire [DATA_WIDTH - 1:0] win_10;
	output wire [DATA_WIDTH - 1:0] win_11;
	output wire [DATA_WIDTH - 1:0] win_12;
	output wire [DATA_WIDTH - 1:0] win_20;
	output wire [DATA_WIDTH - 1:0] win_21;
	output wire [DATA_WIDTH - 1:0] win_22;
	reg [$clog2(TILE_WIDTH + 2) - 1:0] col_cnt;
	reg [$clog2(TILE_HEIGHT + 2) - 1:0] row_cnt;
	wire is_border;
	wire axis_fire;
	wire [DATA_WIDTH - 1:0] core_pixel_in;
	assign is_border = (((row_cnt == 0) || (col_cnt == 0)) || (row_cnt == (TILE_HEIGHT + 1))) || (col_cnt == (TILE_WIDTH + 1));
	assign s_axis_trdy = !is_border && (!m_axis_tvalid || m_axis_trdy);
	assign axis_fire = (is_border && (!m_axis_tvalid || m_axis_trdy)) || ((!is_border && s_axis_tvalid) && s_axis_trdy);
	assign core_pixel_in = (is_border ? {DATA_WIDTH {1'sb0}} : s_axis_tdata);
	always @(posedge clk)
		if (rst) begin
			col_cnt <= 1'sb0;
			row_cnt <= 1'sb0;
		end
		else if (axis_fire) begin
			if (col_cnt == (TILE_WIDTH + 1)) begin
				col_cnt <= 1'sb0;
				if (row_cnt == (TILE_HEIGHT + 1))
					row_cnt <= 1'sb0;
				else
					row_cnt <= row_cnt + 1'b1;
			end
			else
				col_cnt <= col_cnt + 1'b1;
		end
	line_buffer #(
		.DATA_WIDTH(DATA_WIDTH),
		.TILE_WIDTH(TILE_WIDTH + 2)
	) uut(
		.clk(clk),
		.rst(rst),
		.valid_in(axis_fire),
		.pixel_in(core_pixel_in),
		.win_00(win_00),
		.win_01(win_01),
		.win_02(win_02),
		.win_10(win_10),
		.win_11(win_11),
		.win_12(win_12),
		.win_20(win_20),
		.win_21(win_21),
		.win_22(win_22),
		.valid_out(m_axis_tvalid)
	);
endmodule
