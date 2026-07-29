// Top Level AXI wrapper
module tb_line_buffer_top; 
  parameter DATA_WIDTH = 16;
  parameter TILE_WIDTH = 8;
  parameter TILE_HEIGHT = 8;
  parameter TOTAL_PIXELS = TILE_WIDTH * TILE_HEIGHT;
  parameter DELAY      = (2 * TILE_WIDTH) + 2; // 18 cycles latency

  logic clk;
  logic rst;
  logic [DATA_WIDTH-1:0] pixel_in;

  logic [DATA_WIDTH-1:0] win_00, win_01, win_02;
  logic [DATA_WIDTH-1:0] win_10, win_11, win_12;
  logic [DATA_WIDTH-1:0] win_20, win_21, win_22;
  
  logic s_axis_tvalid;
  logic s_axis_trdy;
  logic m_axis_tvalid;
  logic m_axis_trdy;

  // Instantiate Line Buffer
  line_buffer_top #(
    .DATA_WIDTH(DATA_WIDTH),
    .TILE_WIDTH(TILE_WIDTH),
    .TILE_HEIGHT(TILE_HEIGHT)
  ) uut (
    .clk(clk),
    .rst(rst),
    .s_axis_tvalid(s_axis_tvalid),
    .s_axis_trdy(s_axis_trdy),
    .s_axis_tdata(pixel_in),
    .m_axis_tvalid(m_axis_tvalid),
    .m_axis_trdy(m_axis_trdy),
    .win_00(win_00), .win_01(win_01), .win_02(win_02),
    .win_10(win_10), .win_11(win_11), .win_12(win_12),
    .win_20(win_20), .win_21(win_21), .win_22(win_22)
  );

  // 10ns clock generator (100 MHz)
  always #5 clk = ~clk;

  // Feeds the pixels
  initial begin
    pixel_cnt = 1;

    $dumpfile("dump.vcd");
    $dumpvars(0, tb_line_buffer);

    // Initialize inputs
    clk = 0;
    rst = 1;
    s_axis_tvalid = 0;
    pixel_in = 0;
    m_axis_trdy = 1;

    // Apply Reset
    #20;
    rst = 0;
    #10;

    $display("--- STARTING PIXEL STREAM ---");
    s_axis_tvalid = 1'b1;
    pixel_in = pixel_cnt;
    
    while (pixel_cnt <= TOTAL_PIXELS) begin
      @(posedge clk);
      if (s_axis_tvalid && s_axis_trdy) begin
        pixel_cnt++;
        pixel_in <= pixel_cnt; // Non-blocking ensures clean handoff on next cycle
      end
    end
    
    // Stop streaming once all pixels are sent
    s_axis_tvalid = 1'b0; 
  end

  // Watches for windows
  always @(posedge clk) begin
    if (!rst && m_axis_tvalid && m_axis_trdy) begin
      $display("[Time %0t ns] VALID WINDOW PRODUCED!", $time);
      $display("  [%0d] [%0d] [%0d]", win_00, win_01, win_02);
      $display("  [%0d] [%0d] [%0d]", win_10, win_11, win_12);
      $display("  [%0d] [%0d] [%0d]\n", win_20, win_21, win_22);
    end
  end

  // Ensures simulation ends cleanly
  initial begin
    // Hard stop at 2000ns
    #2000;
    $display("--- SIMULATION FINISHED (TIMEOUT REACHED) ---");
    $finish;
  end

endmodule
