#include<cstdio>
#include<iostream>
using namespace std;

template<typename KeyType>
class IntervalTree{
    private:
        //data
        typedef enum Color{RED,BLACK}Color;

        typedef struct IntervalTreeNode{
            IntervalTreeNode *left;
            IntervalTreeNode *right;
            IntervalTreeNode *p;
            Color color;
            KeyType key;
            KeyType low;
            KeyType high;
            KeyType max;
        }*pIntervalTreeNode,IntervalTreeNode;

        pIntervalTreeNode root,nil;

        void destruct(pIntervalTreeNode n){
            if(n==nil) return;
            destruct(n->left);
            destruct(n->right);
            delete n;
        }

        KeyType MAX(KeyType a,KeyType b){
            return a>b? a:b;
        }

        void maxmax(pIntervalTreeNode x){
            if(x!=nil){
                if(x->left!=nil)
                    x->max=MAX(x->high,x->left->max);
                else x->max=x->high;
                if(x->right!=nil)
                    x->max=MAX(x->max,x->right->max);
            }
        }

        void max_up(pIntervalTreeNode x){
            if(x!=nil){
                pIntervalTreeNode y=x->p;
                while(y!=nil){
                    maxmax(y);
                    y=y->p;
                }
            }    
        }

        void left_rotate(pIntervalTreeNode x){
            pIntervalTreeNode y=x->right;
            x->right=y->left;
            if(y->left!=nil)
                y->left->p=x;
            y->p=x->p;
            if(x->p==nil)
                root=y;
            else if(x==x->p->left)
                x->p->left=y;
            else
                x->p->right=y;
            y->left=x;
            x->p=y;

            maxmax(x);
            maxmax(y);
        }

        void right_rotate(pIntervalTreeNode y){
            pIntervalTreeNode x=y->left;
            y->left=x->right;
            if(x->right!=nil)
                x->right->p=y;
            x->p=y->p;
            if(y->p==nil)
                root=x;
            else if(y==y->p->right)
                y->p->right=x;
            else
                y->p->left=x;
            x->right=y;
            y->p=x;

            maxmax(y);
            maxmax(x);
        }

        void insert_fixup(pIntervalTreeNode z){
            //insert fixup
            pIntervalTreeNode x,y;
            while(z->p->color==RED){
                if(z->p==z->p->p->left){
                    y=z->p->p->right;
                    if(y->color==RED){
                        z->p->color=BLACK;
                        y->color=BLACK;
                        z->p->p->color=RED;
                        z=z->p->p;
                    }
                    else{ 
                        if(z==z->p->right){
                            z=z->p;
                            left_rotate(z);
                        }
                        z->p->color=BLACK;
                        z->p->p->color=RED;
                        right_rotate(z->p->p);
                    }
                }
                else{
                    y=z->p->p->left;
                    if(y->color==RED){
                        z->p->color=BLACK;
                        y->color=BLACK;
                        z->p->p->color=RED;
                        z=z->p->p;
                    }
                    else{
                        if(z==z->p->left){
                            z=z->p;
                            right_rotate(z);
                        }
                        z->p->color=BLACK;
                        z->p->p->color=RED;
                        left_rotate(z->p->p);
                    }
                }
            }
            root->color=BLACK;
        }

        void transplant(pIntervalTreeNode u,pIntervalTreeNode v){
            if(u->p==nil)
                root=v;
            else if(u==u->p->left)
                u->p->left=v;
            else
                u->p->right=v;
            v->p=u->p;
            //delete u;
        }

        pIntervalTreeNode minimum(pIntervalTreeNode x){
            while(x->left!=nil)
                x=x->left;
            return x;
        }

        void delete_(pIntervalTreeNode z){
            pIntervalTreeNode x,y;
            y=z;
            Color y_original_color=y->color;
            if(z->left==nil){
                x=z->right;
                transplant(z,z->right);
                maxmax(z->p);
                max_up(z->p);
                delete z;
            }
            else if(z->right==nil){
                x=z->left;
                transplant(z,z->left);
                maxmax(z->p);
                max_up(z->p);
                delete z;
            }
            else{
                y=minimum(z->right);
                y_original_color=y->color;
                x=y->right;//x->p=y;
                if(y->p==z)
                    x->p=y;
                else{
                    transplant(y,y->right);
                    y->right=z->right;
                    y->right->p=y;
                }
                transplant(z,y);
                y->left=z->left;
                y->left->p=y;
                y->color=z->color;
                maxmax(x->p);
                max_up(x->p);
                maxmax(y);
                max_up(y);
                delete z;
            }
            if(y_original_color==BLACK)
                delete_fixup(x);
        }

        void delete_fixup(pIntervalTreeNode x){
            pIntervalTreeNode w;
            while(x!=root && x->color==BLACK){
                if(x==x->p->left){
                    w=x->p->right;
                    if(w->color==RED){
                        w->color=BLACK;
                        x->p->color=RED;
                        left_rotate(x->p);
                        w=x->p->right;
                    }
                    if(w->left->color==BLACK && w->right->color==BLACK){
                        w->color=RED;
                        x=x->p;
                    }
                    else{
                        if(w->right->color==BLACK){
                            w->left->color=BLACK;
                            w->color=RED;
                            right_rotate(w);
                            w=x->p->right;
                        }
                            w->color=x->p->color;
                            x->p->color=BLACK;
                            w->right->color=BLACK;
                            left_rotate(x->p);
                            x=root;
                    }
                    
                }
                else{
                    w=x->p->left;
                    if(w->color==RED){
                        w->color=BLACK;
                        x->p->color=RED;
                        right_rotate(x->p);
                        w=x->p->left;
                    }
                    if(w->left->color==BLACK && w->right->color==BLACK){
                        w->color=RED;
                        x=x->p;
                    }
                    else{
                        if(w->left->color==BLACK){
                            w->right->color=BLACK;
                            w->color=RED;
                            left_rotate(w);
                            w=x->p->left;
                    }
                        w->color=x->p->color;
                        x->p->color=BLACK;
                        w->left->color=BLACK;
                        right_rotate(x->p);
                        x=root;
                    }
                }
            }
            x->color=BLACK;
        }

        void _print(pIntervalTreeNode n,int indent){
            int i;
			if (n == nil) {
                cout<<"<empty tree>"<<endl;
	    		return;
			}
			if (n->right != nil) {
				_print(n->right, indent + 8);
			}
			for(i=0; i<indent; i++)
                cout<<"  ";
			if (n->color == BLACK)
                cout<<"<["<<n->low<<","<<n->high<<"]"<<n->max<<">"<<endl;
			else
                cout<<"["<<n->low<<","<<n->high<<"]"<<n->max<<endl;
			if (n->left != nil) {
					_print(n->left, indent + 8);
			}
        }

        pIntervalTreeNode lookup(KeyType low,KeyType high){
            pIntervalTreeNode n=root;
            while (n != nil && (n->high<low || n->low > high)) {
				if(n->left!=nil && n->left->max >=low)
                    n=n->left;
                else
                    n=n->right;
			}
			return n;
        }

        void _graph(pIntervalTreeNode p,FILE *fp){
            if(p->color==RED)
                fprintf(fp,"<[%.2f,%.2f]%.2f>[fillcolor=red]\n",p->low,p->high,p->max);
                
            if(p->left!=nil){
                fprintf(fp,"<[%.2f,%.2f]%.2f> -> <[%.2f,%.2f]%.2f>\n",p->low,p->high,p->max,
                p->left->low,p->left->high,p->left->max);
                _graph(p->left,fp);
                }
            if(p->right!=nil){
                fprintf(fp,"<[%.2f,%.2f]%.2f> -> <[%.2f,%.2f]%.2f>\n",p->low,p->high,p->max,
                p->right->low,p->right->high,p->right->max);
                _graph(p->right,fp);
            }   
        }

    public:
        //operate
        IntervalTree(){
            nil=new IntervalTreeNode;
            nil->color=BLACK;
            root=nil;
        }

         ~IntervalTree(){
             destruct(root);
             delete nil;    
        }

        void insert(KeyType low,KeyType high){
            pIntervalTreeNode z=new IntervalTreeNode;
            z->low=low;
            z->high=high;
            z->key=low;

            pIntervalTreeNode y=nil,x=root;
            while(x!=nil){
                y=x;
                if(z->key < x->key)
                    x=x->left;
                else
                    x=x->right;
            }
            z->p=y;
            if(y==nil)
                root=z;
            else if(z->key < y->key)
                y->left=z;
            else 
                y->right=z;
            z->left=nil;
            z->right=nil;
            z->color=RED;
            z->max=high;

            max_up(z);
            insert_fixup(z);    
        }

        void print(){
            _print(root,0);
            cout<<"\n\n"<<endl;
        }

        void remove(KeyType low,KeyType high){
            pIntervalTreeNode x=lookup(low,high);
            if(x!=nil)
            {
                cout<<"delete:"<<x->low<<" "<<x->high<<endl;
                delete_(x);
            }
                
        }
        void graph(){
            FILE *fp=fopen("IntervalTree.dot","w");
            fprintf(fp,"digraph g{ \n");
            fprintf(fp,"node [style=filled]\n");
            //fprintf(fp,"%.2f -> %.2f\n",1.2,1.3);
            if(root!=nil){
                fprintf(fp,"<[%.2f,%.2f]%.2f>\n",root->low,root->high,root->max);
                if(root->left!=nil){
                    fprintf(fp,"<[%.2f,%.2f]%.2f> -> <[%.2f,%.2f]%.2f>\n",root->low,root->high,root->max,
                    root->left->low,root->left->high,root->left->max);
                    _graph(root->left,fp);
                }
                if(root->right!=nil){
                    fprintf(fp,"<[%.2f,%.2f]%.2f> -> <[%.2f,%.2f]%.2f>\n",root->low,root->high,root->max,
                    root->right->low,root->right->high,root->right->max);
                    _graph(root->right,fp);
                }
            }
            fprintf(fp,"}");
        }
};



int main(){
    IntervalTree<float> T;

    T.insert(16,21);
    T.insert(8,9);
    T.insert(25,30);
    T.insert(5,8);
    T.insert(15,23);
    T.insert(17,19);
    T.insert(26,26);
    T.insert(0,3);
    T.insert(6,10);
    T.insert(19,20);

    /*
    T.insert(0,3);
    T.insert(6,10);
    T.insert(19,20);
    T.insert(5,8);
    T.insert(15,23);
    T.insert(17,19);
    T.insert(26,26);
    T.insert(8,9);
    T.insert(25,30);
    T.insert(16,21);
    */
    T.print();
    T.remove(17,19);
    
    T.graph();

    //T.remove(16,21);
    //T.remove(16,21);
    //T.remove(16,21);
    //T.remove(16,21);
    //T.remove(16,21);
    /*
    T.remove(1,100);
    T.remove(1,100);
    T.remove(1,100);
    T.remove(1,100);
    T.remove(1,100);
    T.remove(1,100);
    */
    T.print();

}