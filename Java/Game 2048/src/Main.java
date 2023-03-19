
import java.util.Random;
import java.util.Scanner;
    class Game2048 {
        static final int GRID_SIZE = 4;
        static int[][] grid = new int[GRID_SIZE][GRID_SIZE];

        public static void main(String[] args) {
            initializeGrid();
            System.out.println("Welcome to 2048! Use arrow keys to move the tiles. When two tiles with the same number touch, they merge into one.");
            printGrid();

            Scanner sc = new Scanner(System.in);
            while (true) {
                System.out.println("Enter your move (u/d/l/r): ");
                String move = sc.nextLine();
                boolean moved = false;

                if (move.equals("u")) {
                    moved = moveUp();
                } else if (move.equals("d")) {
                    moved = moveDown();
                } else if (move.equals("l")) {
                    moved = moveLeft();
                } else if (move.equals("r")) {
                    moved = moveRight();
                }

                if (moved) {
                    addRandomTile();
                }

                printGrid();

                if (!canMove()) {
                    System.out.println("Game Over!");
                    break;
                }
            }
        }


        private static void initializeGrid() {
            for (int i = 0; i < GRID_SIZE; i++) {
                for (int j = 0; j < GRID_SIZE; j++) {
                    grid[i][j] = 0;
                }
            }
            addRandomTile();
            addRandomTile();
        }

        private static void printGrid() {
            for (int i = 0; i < GRID_SIZE; i++) {
                for (int j = 0; j < GRID_SIZE; j++) {
                    System.out.print(grid[i][j] + " ");
                }
                System.out.println();
            }
        }

        private static boolean moveUp() {
            boolean moved = false;
            for (int j = 0; j < GRID_SIZE; j++) {
                int current = 0;
                int next = current + 1;
                while (next < GRID_SIZE) {
                    while (next < GRID_SIZE && grid[next][j] == 0) {
                        next++;
                    }
                    if (next < GRID_SIZE) {
                        if (grid[current][j] == 0) {
                            grid[current][j] = grid[next][j];
                            grid[next][j] = 0;
                            moved = true;
                        } else if (grid[current][j] == grid[next][j]) {
                            grid[current][j] *= 2;
                            grid[next][j] = 0;
                            current++;
                            moved = true;
                        } else {
                            current = next;
                        }
                        next++;
                    }
                }
            }
            return moved;
        }

        private static boolean moveDown() {
            boolean moved = false;
            for (int j = 0; j < GRID_SIZE; j++) {
                int current = GRID_SIZE - 1;
                int next = current - 1;
                while (next >= 0) {
                    while (next >= 0 && grid[next][j] == 0) {
                        next--;
                    }
                    if (next >= 0) {
                        if (grid[current][j] == 0) {
                            grid[current][j] = grid[next][j];
                            grid[next][j] = 0;
                            moved = true;
                        } else if (grid[current][j] == grid[next][j]) {
                            grid[current][j] *= 2;
                            grid[next][j] = 0;
                            current--;
                            moved = true;
                        } else {
                            current = next;
                        }
                        next--;
                    }
                }
            }
            return moved;
        }

        private static boolean moveLeft() {
            boolean moved = false;

            for (int i = 0; i < GRID_SIZE; i++) {
                int current = 0;
                int next = current + 1;

                while (next < GRID_SIZE) {
                    while (next < GRID_SIZE && grid[i][next] == 0) {
                        next++;
                    }

                    if (next < GRID_SIZE) {
                        if (grid[i][current] == 0) {
                            grid[i][current] = grid[i][next];
                            grid[i][next] = 0;
                            moved = true;
                        } else if (grid[i][current] == grid[i][next]) {
                            grid[i][current] *= 2;
                            grid[i][next] = 0;
                            current++;
                            moved = true;
                        } else {
                            current = next;
                        }
                        next++;
                    }
                }
            }

            return moved;
        }

        private static boolean moveRight() {
            boolean moved = false;

            for (int i = 0; i < GRID_SIZE; i++) {
                int current = GRID_SIZE - 1;
                int next = current - 1;

                while (next >= 0) {
                    while (next >= 0 && grid[i][next] == 0) {
                        next--;
                    }

                    if (next >= 0) {
                        if (grid[i][current] == 0) {
                            grid[i][current] = grid[i][next];
                            grid[i][next] = 0;
                            moved = true;
                        } else if (grid[i][current] == grid[i][next]) {
                            grid[i][current] *= 2;
                            grid[i][next] = 0;
                            current--;
                            moved = true;
                        } else {
                            current = next;
                        }
                        next--;
                    }
                }
            }

            return moved;
        }

        private static void addRandomTile() {
            Random random = new Random();
            int value = random.nextInt(2) * 2 + 2;
            int row, col;
            do {
                row = random.nextInt(GRID_SIZE);
                col = random.nextInt(GRID_SIZE);
            } while (grid[row][col] != 0);
            grid[row][col] = value;
        }

        private static boolean canMove() {
            for (int i = 0; i < GRID_SIZE; i++) {
                for (int j = 0; j < GRID_SIZE; j++) {
                    if (grid[i][j] == 0) {
                        return true;
                    }
                    if (i > 0 && grid[i][j] == grid[i - 1][j]) {
                        return true;
                    }
                    if (i < GRID_SIZE - 1 && grid[i][j] == grid[i + 1][j]) {
                        return true;
                    }
                    if (j > 0 && grid[i][j] == grid[i][j - 1]) {
                        return true;
                    }
                    if (j < GRID_SIZE - 1 && grid[i][j] == grid[i][j + 1]) {
                        return true;
                    }
                }
            }
            return false;
        }
    }