// Top Level AXI wrapper for 3x3 spatial window line buffer.
module line_buffer_top #(
  parameter DATA_WIDTH = 16,
  parameter TILE_WIDTH = 32,
  parameter TILE_HEIGHT = 32)
  (
    input logic clk,
    input logic rst,
    
    // AXI4-Stream Slave Interface
    input logic s_axis_tvalid,
    input logic [DATA_WIDTH-1:0] s_axis_tdata,
    output logic s_axis_trdy,
    
    // AXI4-Stream Master Interface Control
    input logic m_axis_trdy,
    output logic m_axis_tvalid,
    
    output logic [DATA_WIDTH-1:0] win_00, win_01, win_02,
    output logic [DATA_WIDTH-1:0] win_10, win_11, win_12,
    output logic [DATA_WIDTH-1:0] win_20, win_21, win_22
  );
  
  // Internal Counters & Control Logic
  logic [$clog2(TILE_WIDTH + 2) - 1 : 0] col_cnt;
  logic [$clog2(TILE_HEIGHT + 2) - 1 : 0] row_cnt;
  
  logic is_border;
  logic axis_fire;
  logic [DATA_WIDTH-1:0] core_pixel_in;
  
  assign is_border = (row_cnt == 0) || (col_cnt == 0) || (row_cnt == (TILE_HEIGHT + 1)) || (col_cnt == (TILE_WIDTH + 1));
  
  assign s_axis_trdy = !is_border && (!m_axis_tvalid || m_axis_trdy);
  
  assign axis_fire = (is_border && (!m_axis_tvalid || m_axis_trdy)) || (!is_border && s_axis_tvalid && s_axis_trdy);
                                                         
  assign core_pixel_in = is_border ? '0 : s_axis_tdata;                    
  
  // Increments coordinate counters whenever a pixel cycle fires (axis_fire)
  always_ff @(posedge clk) begin
    if (rst) begin
      col_cnt <= '0;
      row_cnt <= '0;
    end
    else if (axis_fire) begin
      if (col_cnt == TILE_WIDTH + 1) begin
        col_cnt <= '0;
        if (row_cnt == TILE_HEIGHT + 1) begin
          row_cnt <= '0;
        end else begin
          row_cnt <= row_cnt + 1'b1;
        end
      end else begin
        col_cnt <= col_cnt + 1'b1;
      end
    end
  end
  
  
  // Instantiate Line Buffer
  // Sized to handle the expanded padded frame width
  line_buffer #(
    .DATA_WIDTH(DATA_WIDTH),
    .TILE_WIDTH(TILE_WIDTH + 2)
  ) uut (
    .clk(clk),
    .rst(rst),
    .valid_in(axis_fire),
    .pixel_in(core_pixel_in),
    .win_00(win_00), .win_01(win_01), .win_02(win_02),
    .win_10(win_10), .win_11(win_11), .win_12(win_12),
    .win_20(win_20), .win_21(win_21), .win_22(win_22),
    .valid_out(m_axis_tvalid)
  );
  
endmodule
