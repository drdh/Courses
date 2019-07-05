#include<cstdio>
#include<iostream>
using namespace std;

template<typename KeyType>
class RBTree{
    private:
        //data
        typedef enum Color{RED,BLACK}Color;

        typedef struct RBTreeNode{
            RBTreeNode *left;
            RBTreeNode *right;
            RBTreeNode *p;
            Color color;
            KeyType key;
        }*pRBTreeNode,RBTreeNode;

        pRBTreeNode root,nil;

        void destruct(pRBTreeNode n){
            if(n==nil) return;
            destruct(n->left);
            destruct(n->right);
            delete n;
        }

        void left_rotate(pRBTreeNode x){
            pRBTreeNode y=x->right;
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
        }

        void right_rotate(pRBTreeNode y){
            pRBTreeNode x=y->left;
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
        }

        void insert_fixup(pRBTreeNode z){
            //insert fixup
            pRBTreeNode x,y;
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

        void transplant(pRBTreeNode u,pRBTreeNode v){
            if(u->p==nil)
                root=v;
            else if(u==u->p->left)
                u->p->left=v;
            else
                u->p->right=v;
            v->p=u->p;
            //delete u;
        }

        pRBTreeNode minimum(pRBTreeNode x){
            while(x->left!=nil)
                x=x->left;
            return x;
        }

        void delete_(pRBTreeNode z){
            pRBTreeNode x,y;
            y=z;
            Color y_original_color=y->color;
            if(z->left==nil){
                x=z->right;
                transplant(z,z->right);
                delete z;
            }
            else if(z->right==nil){
                x=z->left;
                transplant(z,z->left);
                delete z;
            }
            else{
                y=minimum(z->right);
                y_original_color=y->color;
                x=y->right;
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
                delete z;
            }
            if(y_original_color==BLACK)
                delete_fixup(x);
        }

        void delete_fixup(pRBTreeNode x){
            pRBTreeNode w;
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

        void _print(pRBTreeNode n,int indent){
            int i;
			if (n == nil) {
                cout<<"<empty tree>"<<endl;
	    		return;
			}
			if (n->right != nil) {
				_print(n->right, indent + 8);
			}
			for(i=0; i<indent; i++)
                cout<<" ";
			if (n->color == BLACK)
                cout<<"<"<<n->key<<">"<<endl;
			else
                cout<<n->key<<endl;
			if (n->left != nil) {
					_print(n->left, indent + 8);
			}
        }

        pRBTreeNode lookup(KeyType key){
            pRBTreeNode n=root;
            while (n != nil) {
				if (key == n->key) {
					return n;
				} else if (key < n->key) {
					n = n->left;
				} else {
				n = n->right;
				}
			}
			return nil;
        }

        void _graph(pRBTreeNode p,FILE *fp){
            if(p->color==RED){
                fprintf(fp,"%.2f [fillcolor=red]\n",p->key);
            }
            if(p->left!=nil){
                    fprintf(fp,"%.2f -> %.2f\n",p->key,p->left->key);
                    _graph(p->left,fp);
                }
            if(p->right!=nil){
                fprintf(fp,"%.2f -> %.2f\n",p->key,p->right->key);
                _graph(p->right,fp);
            }
        }

    public:
        //operate
        RBTree(){
            nil=new RBTreeNode;
            nil->color=BLACK;
            root=nil;
        }

         ~RBTree(){
             destruct(root);
             delete nil;
        }

        void insert(KeyType key){
            pRBTreeNode z=new RBTreeNode;
            z->key=key;

            pRBTreeNode y=nil,x=root;
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

            insert_fixup(z);
        }

        void print(){
            _print(root,0);
            cout<<"\n\n"<<endl;
        }

        void remove(KeyType key){
            pRBTreeNode x=lookup(key);
            if(x!=nil)
            {
                cout<<"delete:"<<key<<endl;
                delete_(x);
            }

        }

        void graph(){
            FILE *fp=fopen("RBTree.dot","w");
            fprintf(fp,"digraph g{ \n");
            fprintf(fp,"node [style=filled]\n");
            //fprintf(fp,"%.2f -> %.2f\n",1.2,1.3);
            if(root!=nil){
                fprintf(fp,"%.2f\n",root->key);
                if(root->left!=nil){
                    fprintf(fp,"%.2f -> %.2f\n",root->key,root->left->key);
                    _graph(root->left,fp);
                }
                if(root->right!=nil){
                    fprintf(fp,"%.2f -> %.2f\n",root->key,root->right->key);
                    _graph(root->right,fp);
                }
            }
            fprintf(fp,"}");
        }
};


int main(){
    RBTree<float> T;
    T.insert(41);
    T.insert(38);
    T.insert(31);
    T.insert(12);
    T.insert(19);
    //T.insert(8);
    
    T.insert(6);
    T.insert(7);
    T.insert(8);
    T.insert(9);
    T.insert(10);
    T.insert(11);
    //T.insert(12);
    
    T.print();
    T.graph(); 

    T.remove(2);
    T.remove(1);
    T.remove(3);
    T.remove(4);
    T.remove(5);
    T.remove(6);
    T.remove(7);
    T.remove(8);
    
    /*
    T.remove(9);
    T.remove(10);
    T.remove(11);
    T.remove(12);
    T.remove(2);
    */
    T.print();  
}
