import java.util.ArrayList;

class BankAccount {
    private int accountNumber;
    private String accountHolderName;
    private double accountBalance;

    public BankAccount(int accountNumber, String accountHolderName, double accountBalance) {
        this.accountNumber = accountNumber;
        this.accountHolderName = accountHolderName;
        this.accountBalance = accountBalance;
    }

    public int getAccountNumber() {
        return accountNumber;
    }

    public void setAccountNumber(int accountNumber) {
        this.accountNumber = accountNumber;
    }

    public String getAccountHolderName() {
        return accountHolderName;
    }

    public void setAccountHolderName(String accountHolderName) {
        this.accountHolderName = accountHolderName;
    }

    public double getAccountBalance() {
        return accountBalance;
    }

    public void setAccountBalance(double accountBalance) {
        this.accountBalance = accountBalance;
    }

    public void deposit(double amount) {
        accountBalance += amount;
    }

    public void withdraw(double amount) {
        accountBalance -= amount;
    }

    public void displayAccountDetails() {
        System.out.println("Account number: " + accountNumber);
        System.out.println("Account holder name: " + accountHolderName);
        System.out.println("Account balance: " + accountBalance);
    }
}

class SavingAccount extends BankAccount {
    private double interestRate;

    public SavingAccount(int accountNumber, String accountHolderName, double accountBalance, double interestRate) {
        super(accountNumber, accountHolderName, accountBalance);
        this.interestRate = interestRate;
    }

    public double getInterestRate() {
        return interestRate;
    }

    public void setInterestRate(double interestRate) {
        this.interestRate = interestRate;
    }

    public void addInterest() {
        double interest = getAccountBalance() * interestRate / 100;
        deposit(interest);
    }

    @Override
    public void displayAccountDetails() {
        super.displayAccountDetails();
        System.out.println("Interest rate: " + interestRate);
    }
}

class Bank {
    private ArrayList<BankAccount> accounts;
    private int nextAccountNumber;

    public Bank() {
        accounts = new ArrayList<>();
        nextAccountNumber = 1000;
    }

    public void addAccount(BankAccount account) {
        account.setAccountNumber(nextAccountNumber++);
        accounts.add(account);
        System.out.println("Account added successfully!");
    }

    public BankAccount getAccount(int accountNumber) {
        for (BankAccount account : accounts) {
            if (account.getAccountNumber() == accountNumber) {
                return account;
            }
        }
        return null;
    }

    public void displayAllAccounts() {
        for (BankAccount account : accounts) {
            account.displayAccountDetails();
            System.out.println();
        }
    }

    public static void main(String[] args) {
        Bank bank = new Bank();

        BankAccount account1 = new BankAccount(0, "John Doe", 1000);
        bank.addAccount(account1);

        SavingAccount account2 = new SavingAccount(0, "Jane Doe", 2000, 5);
        bank.addAccount(account2);

        account1.deposit(500);
        account2.withdraw(100);

        bank.displayAllAccounts();
    }
}
