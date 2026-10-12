
module conv2d_3x3_mac #(
    parameter DATA_WIDTH = 16
)(
    input wire clk,
    input wire rst_n,
    input wire in_valid,
    output logic out_valid,

    // Channel 0 (IR) 3x3 Window
    input logic signed [DATA_WIDTH-1:0] ch0_win_00, ch0_win_01, ch0_win_02,
    input logic signed [DATA_WIDTH-1:0] ch0_win_10, ch0_win_11, ch0_win_12,
    input logic signed [DATA_WIDTH-1:0] ch0_win_20, ch0_win_21, ch0_win_22,

    // Channel 1 (Radar) 3x3 Window
    input logic signed [DATA_WIDTH-1:0] ch1_win_00, ch1_win_01, ch1_win_02,
    input logic signed [DATA_WIDTH-1:0] ch1_win_10, ch1_win_11, ch1_win_12,
    input logic signed [DATA_WIDTH-1:0] ch1_win_20, ch1_win_21, ch1_win_22,

    output logic signed [DATA_WIDTH-1:0] out_data
);

    logic signed [DATA_WIDTH-1:0] weights [0:17];

    initial begin
        $readmemh("hardware/tb/data/conv1_weights_q015.hex", weights);
    end

    logic signed [2*DATA_WIDTH-1:0] mult_prod [0:17];

    // 18 parallel multipliers for 2 channels of 3x3 convolution
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < 18; i++) begin
                mult_prod[i] <= '0;
            end
        end else if (in_valid) begin
            mult_prod[0]  <= ch0_win_00 * weights[0];
            mult_prod[1]  <= ch0_win_01 * weights[1];
            mult_prod[2]  <= ch0_win_02 * weights[2];
            mult_prod[3]  <= ch0_win_10 * weights[3];
            mult_prod[4]  <= ch0_win_11 * weights[4];
            mult_prod[5]  <= ch0_win_12 * weights[5];
            mult_prod[6]  <= ch0_win_20 * weights[6];
            mult_prod[7]  <= ch0_win_21 * weights[7];
            mult_prod[8]  <= ch0_win_22 * weights[8];

            mult_prod[9]  <= ch1_win_00 * weights[9];
            mult_prod[10] <= ch1_win_01 * weights[10];
            mult_prod[11] <= ch1_win_02 * weights[11];
            mult_prod[12] <= ch1_win_10 * weights[12];
            mult_prod[13] <= ch1_win_11 * weights[13];
            mult_prod[14] <= ch1_win_12 * weights[14];
            mult_prod[15] <= ch1_win_20 * weights[15];
            mult_prod[16] <= ch1_win_21 * weights[16];
            mult_prod[17] <= ch1_win_22 * weights[17];
        end
    end

    // Sum the products to get the final output
    logic signed [2*DATA_WIDTH:0] sum_l1 [0:8];
    logic signed [(2*DATA_WIDTH+1):0] sum_l2 [0:4];
    logic signed [(2*DATA_WIDTH+1):0] sum_l3 [0:2];
    logic signed [(2*DATA_WIDTH+1):0] final_sum;

    always_comb begin
        // Level 1: Sum pairs of products
        sum_l1[0] = mult_prod[0] + mult_prod[1];
        sum_l1[1] = mult_prod[2] + mult_prod[3];
        sum_l1[2] = mult_prod[4] + mult_prod[5];
        sum_l1[3] = mult_prod[6] + mult_prod[7];
        sum_l1[4] = mult_prod[8] + mult_prod[9];
        sum_l1[5] = mult_prod[10] + mult_prod[11];
        sum_l1[6] = mult_prod[12] + mult_prod[13];
        sum_l1[7] = mult_prod[14] + mult_prod[15];
        sum_l1[8] = mult_prod[16] + mult_prod[17];

        // Level 2: Sum pairs of sums from level 1
        sum_l2[0] = sum_l1[0] + sum_l1[1];
        sum_l2[1] = sum_l1[2] + sum_l1[3];
        sum_l2[2] = sum_l1[4] + sum_l1[5];
        sum_l2[3] = sum_l1[6] + sum_l1[7];
        sum_l2[4] = sum_l1[8];

        // Level 3: Sum pairs of sums from level 2
        sum_l3[0] = sum_l2[0] + sum_l2[1];
        sum_l3[1] = sum_l2[2] + sum_l2[3];
        sum_l3[2] = sum_l2[4];

        // Final Sum
        final_sum = sum_l3[0] + sum_l3[1] + sum_l3[2];

    end

    logic signed [(2*DATA_WIDTH+1):0] shifted_sum;

    always_comb begin
        shifted_sum = final_sum >>> 15; // Right shift by 15 for Q1.15 format
    end

    logic v_stage1;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v_stage1 <= 1'b0;
            out_data <= '0;
            out_valid <= 1'b0;
        end else begin
            v_stage1 <= in_valid;
            out_valid <= v_stage1;

            if (shifted_sum > 32767) begin
                out_data <= 16'sh7FFF; // Saturate to max positive value
            end else if (shifted_sum < -32768) begin
                out_data <= 16'sh8000; // Saturate to max negative value
            end else begin
                out_data <= shifted_sum[DATA_WIDTH-1:0]; // Assign the lower DATA_WIDTH bits
            end
        end
    end

endmodule