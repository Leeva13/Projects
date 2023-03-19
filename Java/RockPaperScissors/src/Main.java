import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.println("Enter your choice: Rock (r), Paper (p), or Scissors (s)");
        String userChoice = sc.nextLine();

        int computerChoice = (int) (Math.random() * 3) + 1;
        String computer = "";
        if (computerChoice == 1) {
            computer = "r";
        } else if (computerChoice == 2) {
            computer = "p";
        } else if (computerChoice == 3) {
            computer = "s";
        }

        System.out.println("Computer chooses: " + computer);

        if (userChoice.equals(computer)) {
            System.out.println("It's a tie!");
        } else if (userChoice.equals("r")) {
            if (computer.equals("s")) {
                System.out.println("You win!");
            } else {
                System.out.println("You lose!");
            }
        } else if (userChoice.equals("p")) {
            if (computer.equals("r")) {
                System.out.println("You win!");
            } else {
                System.out.println("You lose!");
            }
        } else if (userChoice.equals("s")) {
            if (computer.equals("p")) {
                System.out.println("You win!");
            } else {
                System.out.println("You lose!");
            }
        }
    }
}