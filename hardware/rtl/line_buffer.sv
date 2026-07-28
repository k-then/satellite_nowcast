// Converts a 1D pixel stream into a continuous 3x3 spatial window
// using circular RAM buffers and horizontal shift registers.

module line_buffer #(parameter DATA_WIDTH = 16, parameter TILE_WIDTH = 32) 
  (
    input logic clk, rst, valid_in, 
    input logic [DATA_WIDTH - 1:0] pixel_in, 
    output logic [DATA_WIDTH - 1:0] win_00, win_01, win_02,
    output logic [DATA_WIDTH - 1:0] win_10, win_11, win_12,
    output logic [DATA_WIDTH - 1:0] win_20, win_21, win_22,
    output logic valid_out
  );
  
  
  // Internal Line Buffers (On-Chip RAMs)
  // Store the previous two horizontal pixel rows for vertical window alignment
  logic [DATA_WIDTH - 1:0] line_ram_0 [0:TILE_WIDTH - 1];
  logic [DATA_WIDTH - 1:0] line_ram_1 [0:TILE_WIDTH - 1];
  
  // Pointers and Latency Parameters
  logic [$clog2(TILE_WIDTH)-1:0] ptr;
  
  // Pipeline delay before first valid 3x3 window
  localparam delay = (2 * TILE_WIDTH) + 2;
  logic [$clog2(delay + 1) - 1:0] pixel_count;
  
  // Internal 3x3 Window Matrix Register
  logic [DATA_WIDTH-1:0] win [0:2][0:2];
  
  assign win_00 = win[0][0]; assign win_01 = win[0][1]; assign win_02 = win[0][2];
  assign win_10 = win[1][0]; assign win_11 = win[1][1]; assign win_12 = win[1][2];
  assign win_20 = win[2][0]; assign win_21 = win[2][1]; assign win_22 = win[2][2];
  
  // Sequential Processing Pipeline
  always_ff @(posedge clk) begin
    if (rst) begin
      ptr <= '0;
  	  pixel_count <= '0;
  	  valid_out <= 1'b0;
      
      win[0][0] <= '0; win[0][1] <= '0; win[0][2] <= '0;
      win[1][0] <= '0; win[1][1] <= '0; win[1][2] <= '0;
      win[2][0] <= '0; win[2][1] <= '0; win[2][2] <= '0;
      
    end else if (valid_in) begin
      
      // Horizontal Shift Register: Shifts existing columns left by 1 position
      win[0][0] <= win[0][1];  win[0][1] <= win[0][2];
	  win[1][0] <= win[1][1];  win[1][1] <= win[1][2];
	  win[2][0] <= win[2][1];  win[2][1] <= win[2][2];
      
      // Vertical Taps: Loads the new column from RAMs and input stream
      win[0][2] <= line_ram_1[ptr];
      win[1][2] <= line_ram_0[ptr];
      win[2][2] <= pixel_in;
      
      // RAM Cascading: Pushes pixels down through memory 
      line_ram_1[ptr] <= line_ram_0[ptr];
      line_ram_0[ptr] <= pixel_in;
      
      ptr <= (ptr == TILE_WIDTH - 1) ? '0 : ptr + 1'b1;
      
      // Latency Tracking
      if (pixel_count < delay) begin
    	pixel_count <= pixel_count + 1'b1;
    	valid_out   <= 1'b0;
	  end else begin
    	valid_out   <= 1'b1;
	  end
      
    end else begin
      valid_out <= 1'b0;
    end
    
  end
  
endmodule