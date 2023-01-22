import java.util.Scanner;
public class Main {

    public static class TicTacToe {
        private char[][] board;
        private char currentPlayerMark;

        public TicTacToe() {
            board = new char[3][3];
            currentPlayerMark = 'x';
            initializeBoard();
        }

        public void initializeBoard() {
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    board[i][j] = '-';
                }
            }
        }

        public void printBoard() {
            System.out.println("-------------");
            for (int i = 0; i < 3; i++) {
                System.out.print("| ");
                for (int j = 0; j < 3; j++) {
                    System.out.print(board[i][j] + " | ");
                }
                System.out.println();
                System.out.println("-------------");
            }
        }

        public boolean isBoardFull() {
            boolean isFull = true;
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    if (board[i][j] == '-') {
                        isFull = false;
                    }
                }
            }
            return isFull;
        }

        public boolean checkForWin() {
            return (checkRowsForWin() || checkColumnsForWin() || checkDiagonalsForWin());
        }

        private boolean checkRowsForWin() {
            for (int i = 0; i < 3; i++) {
                if (checkRowCol(board[i][0], board[i][1], board[i][2]) == true) {
                    return true;
                }
            }
            return false;
        }

        private boolean checkColumnsForWin() {
            for (int i = 0; i < 3; i++) {
                if (checkRowCol(board[0][i], board[1][i], board[2][i]) == true) {
                    return true;
                }
            }
            return false;
        }

        private boolean checkDiagonalsForWin() {
            return ((checkRowCol(board[0][0], board[1][1], board[2][2]) == true) || (checkRowCol(board[0][2], board[1][1], board[2][0]) == true));
        }

        private boolean checkRowCol(char c1, char c2, char c3) {
            return ((c1 != '-') && (c1 == c2) && (c2 == c3));
        }

        public void changePlayer() {
            if (currentPlayerMark == 'x') {
                currentPlayerMark = 'o';
            } else {
                currentPlayerMark = 'x';
            }
        }

        public boolean placeMark(int row, int col) {
            if ((row >= 0) && (row < 3)) {
                if ((col >= 0) && (col < 3)) {
                    if (board[row][col] == '-') {
                        board[row][col] = currentPlayerMark;
                        return true;
                    }
                }
            }
            return false;
        }
    }

    public static void main(String[] args) {
        TicTacToe game = new TicTacToe();
        Scanner scanner = new Scanner(System.in);

        while (game.checkForWin() == false && game.isBoardFull() == false) {
            System.out.println("Current player: " + game.currentPlayerMark);
            System.out.println("Enter a row and column: ");
            int row = scanner.nextInt() - 1;
            int col = scanner.nextInt() - 1;
            if (game.placeMark(row, col)) {
                game.printBoard();
                if (game.checkForWin() == true) {
                    System.out.println("Current player " + game.currentPlayerMark + " wins!");
                } else if (game.isBoardFull() == true) {
                    System.out.println("It's a tie!");
                }
                game.changePlayer();
            } else {
                System.out.println("Invalid Input. Please enter row and column value between 1-3");
            }
        }
        scanner.close();
    }
}