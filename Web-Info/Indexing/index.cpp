#include<iostream>
#include<vector>
#include<string>
#include<fstream>
#include<sstream>
#include<algorithm>
#include<map>
#include<set>
#include<cstdlib>

using namespace std;

vector<string> &split( const string &str, const string &delimiters, vector<string> &elems, bool skip_empty = true ) {
    string::size_type pos, prev = 0;
    while ( ( pos = str.find_first_of(delimiters, prev) ) != string::npos ) {
        if ( pos > prev ) {
            if ( skip_empty && 1 == pos - prev ) break;
            elems.emplace_back( str, prev, pos - prev );
        }
        prev = pos + 1;
    }
    if ( prev < str.size() ) elems.emplace_back( str, prev, str.size() - prev );
    return elems;
}

bool compare(const pair<string,int>&a,const pair<string,int>&b)
{
    return a.first<=b.first;
}

int main(int argc,char *argv[])
{
    ifstream in;   
    vector<pair<string,int>> termDoc;

    int N;
    if(argc!=2)
    {
        cout<<"usage: make test n=4\n"<<"default n=4\n"<<endl;
        N=4;
    }  
    else N=atoi(argv[1]);

    for(int i=1;i<=N;i++)
    {
        in.open("./doc/"+to_string(i)+".txt");
        ostringstream tmp;
        tmp << in.rdbuf();
        in.close();
        string str = tmp.str();

        vector<string> result;
        for ( const auto &s : split( str, ",.?!; \n\t\\<>{}[]()|/", result ) )
        {
            //cout << s << s.length()<<" ";
            termDoc.push_back(pair<string,int>(s,i));
        }    
        //cout << std;  
    }

    sort(termDoc.begin(),termDoc.end(),compare);

    map<string,set<int>>PostList;
    for(auto it=termDoc.begin();it!=termDoc.end();it++)
    {
        //cout<<(*it).first<<" "<<(*it).second<<endl;
        PostList[(*it).first].insert((*it).second);
    }    
    
    ofstream out("index.txt");
    for(auto i=PostList.begin();i!=PostList.end();i++)
    {
        out<<(*i).first<<" ";
        for(auto j=(i->second).begin();j!=(i->second).end();j++)
        {
            out<<*j<<" ";
        }
        out<<endl;
    }
    out.close();
    
    ofstream out1("dict.txt");
    ofstream out2("list.txt");
    for(auto i=PostList.begin();i!=PostList.end();i++)
    {
        out1<<(*i).first<<" "<<out2.tellp()<<endl;       
            out2<<(i->second).size()<<" ";
        for(auto j=(i->second).begin();j!=(i->second).end();j++)
        {
            out2<<*j<<" ";
        }
    }
    out2<<endl;
    out1.close();
    out2.close();
}

