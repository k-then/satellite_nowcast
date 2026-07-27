// Acts as a camera sensor that sends pixels into the line buffer and checks the line matrix
module tb_line_buffer;

  // Use smaller tile (8x8) for quick human inspection
  parameter DATA_WIDTH = 16;
  parameter TILE_WIDTH = 8;
  parameter DELAY      = (2 * TILE_WIDTH) + 2; // 18 cycles latency

  logic clk;
  logic rst;
  logic valid_in;
  logic [DATA_WIDTH-1:0] pixel_in;

  logic [DATA_WIDTH-1:0] win_00, win_01, win_02;
  logic [DATA_WIDTH-1:0] win_10, win_11, win_12;
  logic [DATA_WIDTH-1:0] win_20, win_21, win_22;
  logic valid_out;

  // Instantiate Line Buffer
  line_buffer #(
    .DATA_WIDTH(DATA_WIDTH),
    .TILE_WIDTH(TILE_WIDTH)
  ) uut (
    .clk(clk),
    .rst(rst),
    .valid_in(valid_in),
    .pixel_in(pixel_in),
    .win_00(win_00), .win_01(win_01), .win_02(win_02),
    .win_10(win_10), .win_11(win_11), .win_12(win_12),
    .win_20(win_20), .win_21(win_21), .win_22(win_22),
    .valid_out(valid_out)
  );

  // 10ns clock generator (100 MHz)
  always #5 clk = ~clk;

  initial begin
    // Setup waveform dump file
    $dumpfile("dump.vcd");
    $dumpvars(0, tb_line_buffer);

    // Initialize inputs
    clk      = 0;
    rst      = 1;
    valid_in = 0;
    pixel_in = 0;

    // Apply Reset
    #20;
    rst = 0;
    #10;

    $display("--- STARTING PIXEL STREAM ---");
    for (int i = 1; i <= 70; i++) begin
      @(posedge clk);
      valid_in <= 1'b1;
      pixel_in <= i;

      // Prints output state once valid_out goes HIGH
      if (valid_out) begin
        $display("[Time %0t ns] VALID WINDOW PRODUCED!", $time);
        $display("  [%0d] [%0d] [%0d]", win_00, win_01, win_02);
        $display("  [%0d] [%0d] [%0d]", win_10, win_11, win_12);
        $display("  [%0d] [%0d] [%0d]\n", win_20, win_21, win_22);
      end
    end

    @(posedge clk);
    valid_in <= 1'b0;
    #50;

    $display("--- SIMULATION FINISHED ---");
    $finish;
  end

endmodule