int checkPrimeNumber(int number)
{
    int divisor, flag = 1;    
    if (number < 1) flag = 0;  //bad: the callee should not verify parameter
    for(divisor=2; divisor <= number/2; ++divisor)
    {
        if (number%divisor == 0)
        {
            flag =0;
            break;
        }
    }
    return flag;
}

int main()
{
    int number=10, flag;
    // need to be checked
    flag = checkPrimeNumber(number);
    return 0;
}

