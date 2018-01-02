public class TestEnum {
    public enum MyColor {
        red,green,yellow
    };
    public static void main(String[] args) {
        MyColor m = MyColor.red;
        switch (m) {
            case red :
                System.out.println("red");
                break ;
            case yellow:
                System.out.println("yellow");
                break;
            case green:
                System.out.println("green");
                break;
        }
    }
}
