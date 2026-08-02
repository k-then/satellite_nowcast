// Top Level AXI wrapper for 3x3 spatial window line buffer.
module line_buffer_top #(
  parameter int DATA_WIDTH = 16,
  parameter int TILE_WIDTH = 32,
  parameter int TILE_HEIGHT = 32,
  parameter int CHANNELS = 2
  )
  (
    input logic clk,
    input logic rst,
    
    // AXI4-Stream Slave Interface
    // [31:16] = Channel 1 (Radar), [15:0] = Channel 0 (IR)
    input logic s_axis_tvalid,
    input logic [CHANNELS * DATA_WIDTH-1:0] s_axis_tdata,
    input logic s_axis_tlast,
    output logic s_axis_trdy,
    
    // AXI4-Stream Master Interface Control
    input logic m_axis_trdy,
    output logic m_axis_tvalid,
    output logic m_axis_tlast,
    
    // Channel 0 (IR) 3x3 Window
    output logic [DATA_WIDTH-1:0] ch0_win_00, ch0_win_01, ch0_win_02,
    output logic [DATA_WIDTH-1:0] ch0_win_10, ch0_win_11, ch0_win_12,
    output logic [DATA_WIDTH-1:0] ch0_win_20, ch0_win_21, ch0_win_22,

    // Channel 1 (Radar) 3x3 Window
    output logic [DATA_WIDTH-1:0] ch1_win_00, ch1_win_01, ch1_win_02,
    output logic [DATA_WIDTH-1:0] ch1_win_10, ch1_win_11, ch1_win_12,
    output logic [DATA_WIDTH-1:0] ch1_win_20, ch1_win_21, ch1_win_22
  );
  
  // Define bit-width parameters before using them in vector declarations
  localparam int ROW_BITS = $clog2(TILE_HEIGHT + 2);
  localparam int COL_BITS = $clog2(TILE_WIDTH + 2);

  // Internal Counters using named width parameters
  logic [COL_BITS-1:0] col_cnt;
  logic [ROW_BITS-1:0] row_cnt;


  typedef enum logic [1:0] {IDLE, STREAMING, FLUSHING} state_t;
  state_t state_reg, state_next;

  localparam int unsigned FLUSH_CYCLES = 3 * (TILE_WIDTH + 2) + 5;
  localparam int unsigned FLUSH_BITS   = $clog2(FLUSH_CYCLES);
  localparam logic [FLUSH_BITS-1:0] FLUSH_TARGET = FLUSH_BITS'(FLUSH_CYCLES - 1);
  logic [FLUSH_BITS-1:0] flush_cnt;
  
  logic is_border;
  logic axis_fire;

  logic [CHANNELS-1:0][DATA_WIDTH-1:0] s_axis_tdata_split;
  logic [CHANNELS-1:0][DATA_WIDTH-1:0] core_pixel_in;

  assign s_axis_tdata_split = s_axis_tdata;
  
  assign is_border = (row_cnt == 0) || (col_cnt == 0) || (row_cnt == ROW_BITS'(TILE_HEIGHT + 1)) || (col_cnt == COL_BITS'(TILE_WIDTH + 1));

  // s_axis_trdy is active during streaming/idle when downstream can accept pixels and not padded border
  assign s_axis_trdy = (state_reg != FLUSHING) && !is_border && (!m_axis_tvalid || m_axis_trdy);
  
  assign axis_fire = (state_reg == FLUSHING) ? (!m_axis_tvalid || m_axis_trdy) : (is_border) ? (!m_axis_tvalid || m_axis_trdy) :(s_axis_tvalid && s_axis_trdy);
  
  // Zero-padding logic applied across both channels on border pixels
  genvar c_in;
  generate
    for (c_in = 0; c_in < CHANNELS; c_in++) begin : g_pad
      assign core_pixel_in[c_in] = (is_border || state_reg == FLUSHING) ? '0 : s_axis_tdata_split[c_in];
    end
  endgenerate
              
  
  // Increments coordinate counters whenever a pixel cycle fires (axis_fire)
  always_ff @(posedge clk) begin
    if (rst) begin
      col_cnt <= '0;
      row_cnt <= '0;
    end
    else if (axis_fire) begin
      if (col_cnt == COL_BITS'(TILE_WIDTH + 1)) begin
        col_cnt <= '0;
        if (row_cnt == ROW_BITS'(TILE_HEIGHT + 1)) begin
          row_cnt <= '0;
        end else begin
          row_cnt <= row_cnt + 1'b1;
        end
      end else begin
        col_cnt <= col_cnt + 1'b1;
      end
    end
  end
  

  // FF Block for FSM States
  always_ff @(posedge clk) begin
    if (rst) begin
      state_reg <= IDLE;
      flush_cnt <= '0;
    end else begin
      state_reg <= state_next;
      if (state_reg == FLUSHING) begin
        if (axis_fire) flush_cnt <= flush_cnt + 1'b1;
      end else begin
        flush_cnt <= '0;
      end
    end
  end

  // Next State Logic for pixel processing
  always_comb begin 
    state_next = state_reg;
    case (state_reg)

      IDLE, STREAMING: begin
        if (s_axis_tvalid && s_axis_trdy && s_axis_tlast) begin
          state_next = FLUSHING;
        end else if (s_axis_tvalid && s_axis_trdy) begin
          state_next = STREAMING;
        end
      end

      FLUSHING: begin
        if (axis_fire && (flush_cnt == FLUSH_TARGET)) begin
          state_next = IDLE;
        end
      end
      default: state_next = IDLE;
    endcase
  end

  // Array signals to capture windows from the generate loop
  logic [CHANNELS-1:0][DATA_WIDTH-1:0] win_00, win_01, win_02;
  logic [CHANNELS-1:0][DATA_WIDTH-1:0] win_10, win_11, win_12;
  logic [CHANNELS-1:0][DATA_WIDTH-1:0] win_20, win_21, win_22;
  /* verilator lint_off UNUSEDSIGNAL */
  logic [CHANNELS-1:0] ch_valid_out;
  /* verilator lint_on UNUSEDSIGNAL */

  // Instantiate Parallel Line Buffers for Each Channel
  genvar c;
  generate
    for (c = 0; c < CHANNELS; c++) begin : g_line_buffers
      line_buffer #(
        .DATA_WIDTH(DATA_WIDTH),
        .TILE_WIDTH(TILE_WIDTH + 2)
      ) u_line_buf (
        .clk(clk),
        .rst(rst),
        .valid_in(axis_fire),
        .pixel_in(core_pixel_in[c]),
        .win_00(win_00[c]), .win_01(win_01[c]), .win_02(win_02[c]),
        .win_10(win_10[c]), .win_11(win_11[c]), .win_12(win_12[c]),
        .win_20(win_20[c]), .win_21(win_21[c]), .win_22(win_22[c]),
        .valid_out(ch_valid_out[c])
      );
    end
  endgenerate

  // Output valid is synchronous across channels (use Channel 0 valid)
  assign m_axis_tvalid = ch_valid_out[0];

  assign m_axis_tlast = m_axis_tvalid && (state_reg == FLUSHING) && (flush_cnt == FLUSH_TARGET);

  // Map internal channel array outputs to explicit top-level ports
  assign ch0_win_00 = win_00[0]; assign ch0_win_01 = win_01[0]; assign ch0_win_02 = win_02[0];
  assign ch0_win_10 = win_10[0]; assign ch0_win_11 = win_11[0]; assign ch0_win_12 = win_12[0];
  assign ch0_win_20 = win_20[0]; assign ch0_win_21 = win_21[0]; assign ch0_win_22 = win_22[0];

  assign ch1_win_00 = win_00[1]; assign ch1_win_01 = win_01[1]; assign ch1_win_02 = win_02[1];
  assign ch1_win_10 = win_10[1]; assign ch1_win_11 = win_11[1]; assign ch1_win_12 = win_12[1];
  assign ch1_win_20 = win_20[1]; assign ch1_win_21 = win_21[1]; assign ch1_win_22 = win_22[1];
  
endmodule
