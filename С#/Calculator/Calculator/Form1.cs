using System.Data;

namespace Calculator {
    public partial class Form1 : Form {

        private string currentCalculation = "";

        public Form1() {
            InitializeComponent();
        }

        private void button1_Click(object sender, EventArgs e) {

        }
        private void button2_Click(object sender, EventArgs e) {

        }
        private void button3_Click(object sender, EventArgs e) {

        }
        private void textBox1_TextChanged(object sender, EventArgs e) {

        }
        private void button_Click(object sender, EventArgs e) {
            // This adds the number or operator to the string calculation
            currentCalculation += (sender as Button).Text;

            // Display the current calculation back to the user
            textBoxOutput.Text = currentCalculation;
        }
        private void button_Equals_Click(object sender, EventArgs e) {
            string formattedCalculation = currentCalculation.ToString().Replace("X", "*").ToString().Replace("&divide;", "/");

            try {
                textBoxOutput.Text = new DataTable().Compute(formattedCalculation, null).ToString();
                currentCalculation = textBoxOutput.Text;
            }
            catch (Exception ex) {
                textBoxOutput.Text = "0";
                currentCalculation = "";
            }
        }
        private void button_Clear_Click(object sender, EventArgs e) {
            // Reset the calculation and empty the textbox
            textBoxOutput.Text = "0";
            currentCalculation = "";
        }
    }
}