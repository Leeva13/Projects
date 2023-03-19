import java.util.ArrayList;

class Library {
    private ArrayList<Book> books;
    private ArrayList<Member> members;

    public Library() {
        books = new ArrayList<>();
        members = new ArrayList<>();
    }

    public void addBook(Book book) {
        books.add(book);
    }

    public void addMember(Member member) {
        members.add(member);
    }

    public void issueBook(Book book, Member member) {
        if (books.contains(book) && members.contains(member)) {
            if (!book.isIssued()) {
                book.setIssued(true);
                member.borrowBook(book);
                System.out.println("Book issued to " + member.getName());
            } else {
                System.out.println("Book is already issued");
            }
        } else {
            System.out.println("Either book or member does not exist");
        }
    }

    public void returnBook(Book book, Member member) {
        if (books.contains(book) && members.contains(member)) {
            if (book.isIssued() && member.hasBook(book)) {
                book.setIssued(false);
                member.returnBook(book);
                System.out.println("Book returned by " + member.getName());
            } else {
                System.out.println("Book is not issued or not borrowed by this member");
            }
        } else {
            System.out.println("Either book or member does not exist");
        }
    }
}

class Book {
    private String title;
    private String author;
    private boolean isIssued;

    public Book(String title, String author) {
        this.title = title;
        this.author = author;
        this.isIssued = false;
    }

    public String getTitle() {
        return title;
    }

    public String getAuthor() {
        return author;
    }

    public boolean isIssued() {
        return isIssued;
    }

    public void setIssued(boolean issued) {
        isIssued = issued;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Book book = (Book) o;
        return title.equals(book.title) &&
                author.equals(book.author);
    }
}

class Member {
    private String name;
    private ArrayList<Book> borrowedBooks;

    public Member(String name) {
        this.name = name;
        borrowedBooks = new ArrayList<>();
    }

    public String getName() {
        return name;
    }

    public void borrowBook(Book book) {
        borrowedBooks.add(book);
    }

    public void returnBook(Book book) {
        borrowedBooks.remove(book);
    }

    public boolean hasBook(Book book) {
        return borrowedBooks.contains(book);
    }


    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Member member = (Member) o;
        return name.equals(member.name);
    }


}
public class Main {
    public static void main(String[] args) {
        Library library = new Library();

        Book book1 = new Book("The Great Gatsby", "F. Scott Fitzgerald");
        Book book2 = new Book("To Kill a Mockingbird", "Harper Lee");
        library.addBook(book1);
        library.addBook(book2);

        Member member1 = new Member("John Doe");
        Member member2 = new Member("Jane Doe");
        library.addMember(member1);
        library.addMember(member2);

        library.issueBook(book1, member1);
        library.issueBook(book2, member2);
        library.returnBook(book1, member1);
        library.returnBook(book2, member2);
    }
}