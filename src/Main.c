#include "/home/codeleaded/System/Static/Library/WindowEngine1.0.h"
#include "/home/codeleaded/System/Static/Library/NeuralNetwork.h"
#include "/home/codeleaded/System/Static/Library/Thread.h"

#define PADDLE_1_POSX       0.05f
#define PADDLE_2_POSX       0.95f
#define PADDLE_WIDTH        0.025f
#define PADDLE_HEIGHT       0.18f
#define PADDLE_SPEED        0.50f

#define BALL_WIDTH          0.0125f
#define BALL_HEIGHT         0.025f
#define BALL_STARTSPEED     -0.1f
#define SPEED_ACCSPEED      0.05f

#define PADDLE_1_COLOR      BLUE
#define PADDLE_2_COLOR      RED
#define BALL_COLOR          WHITE

#define NN_LEARNRATE        0.5f
#define NN_DELAY            1000
#define NN_TICKSUPDATE      10


typedef struct PongObject{
    Vec2 p;
    Vec2 d;
    Vec2 v;
    Pixel c;
} PongObject;

PongObject paddle1;
PongObject paddle2;
PongObject ball;
int score1 = 0;
int score2 = 0;
NeuralRewardType aireward = 0.0f;

void Reset(){
    paddle1.p.y = 0.5f - PADDLE_HEIGHT * 0.5f;
    paddle2.p.y = 0.5f - PADDLE_HEIGHT * 0.5f;
    
    ball.p.x = 0.5f - BALL_WIDTH * 0.5f;
    ball.p.y = 0.5f - BALL_HEIGHT * 0.5f;
    //ball.v.x = BALL_STARTSPEED;
    //ball.v.y = 0.0f;

    float a = Random_f64_MinMax(0.0f,2.0f * F32_PI);
    a = F32_Border(a,0.25f * F32_PI,0.75f * F32_PI);
    a = F32_Border(a,1.25f * F32_PI,1.75f * F32_PI);

    ball.v = Vec2_Mulf(Vec2_OfAngle(a),BALL_STARTSPEED);
}

PongObject PongObject_New(Vec2 p,Vec2 d,Vec2 v,Pixel c){
    PongObject po;
    po.p = p;
    po.d = d;
    po.v = v;
    po.c = c;
    return po;
}
void PongObject_Update(PongObject* po,float t){
    po->p = Vec2_Add(po->p,Vec2_Mulf(po->v,t));

    if(po->p.x < -po->d.x){
        score2++;
        Reset();
    }
    if(po->p.x > 1.0f){
        score1++;
        Reset();
    }
    if(po->p.y < 0.0f){
        po->p.y = 0.0f;
        po->v.y *= -1.0f;
    }
    if(po->p.y > 1.0f - po->d.y){
        po->p.y = 1.0f - po->d.y;
        po->v.y *= -1.0f;
    }
}
void PongObject_Collision(PongObject* po,PongObject* target){
    if(Overlap_Rect_Rect((Rect){ po->p,po->d },(Rect){ target->p,target->d })){
        float s = Vec2_Mag(target->v);
        target->v.x *= -1.0f;
        target->v = Vec2_Mulf(Vec2_Norm(target->v),s + SPEED_ACCSPEED);
        target->v.y += po->v.y;
    }
}
void PongObject_Render(PongObject* po){
    RenderRect(po->p.x * GetWidth(),po->p.y * GetHeight(),po->d.x * GetWidth(),po->d.y * GetHeight(),po->c);
}

void PongObject_Step_Update(PongObject* po,float t){
    po->p = Vec2_Add(po->p,Vec2_Mulf(po->v,t));

    if(po->p.x < -po->d.x){
        aireward += 10.0f;
    }
    if(po->p.x > 1.0f){
        aireward -= 5.0f;
    }
    if(po->p.y < 0.0f){
        po->p.y = 0.0f;
        po->v.y *= -1.0f;
    }
    if(po->p.y > 1.0f - po->d.y){
        po->p.y = 1.0f - po->d.y;
        po->v.y *= -1.0f;
    }
}
void PongObject_Step_Collision(PongObject* po,PongObject* target){
    if(Overlap_Rect_Rect((Rect){ po->p,po->d },(Rect){ target->p,target->d })){
        float s = Vec2_Mag(target->v);
        target->v.x *= -1.0f;
        target->v = Vec2_Mulf(Vec2_Norm(target->v),s + SPEED_ACCSPEED);
        target->v.y += po->v.y;

        if(po == &paddle2){
            aireward += 1.0f;
        }
    }
}


typedef struct PongInfos{
    Vec2 paddle1_p;
    Vec2 paddle1_v;
    Vec2 paddle2_p;
    Vec2 paddle2_v;
    Vec2 ball_p;
    Vec2 ball_v;
} PongInfos;

#define TIME_STEP   0.10f

float ElapTime = 0.05f;
NeuralNetwork neuralnet;
AlxFont infofont;
Thread updaterai;

NeuralRewardType NeuralEnviroment_Func_Step(NeuralEnviroment* ne,int d){
    NeuralRewardType reward = aireward;
    aireward = 0.0f;

    if(d==0)      paddle2.v.y = -PADDLE_SPEED;
    else if(d==1) paddle2.v.y = PADDLE_SPEED;
    else if(d==2) paddle2.v.y = 0.0f;
    
    PongObject_Step_Update(&paddle1,TIME_STEP);
    PongObject_Step_Update(&paddle2,TIME_STEP);
    PongObject_Step_Update(&ball,TIME_STEP);

    PongObject_Step_Collision(&paddle1,&ball);
    PongObject_Step_Collision(&paddle2,&ball);

    NeuralRewardType ret = aireward;
    aireward = reward;
    return ret;
}
NeuralEnviromentBlock NeuralEnviroment_Func_Block(NeuralEnviroment* ne){
    NeuralEnviromentBlock neb = NeuralEnviromentBlock_New((PongInfos[]){{
        paddle1.p,paddle1.v,
        paddle2.p,paddle2.v,
        ball.p,ball.v
    }},sizeof(PongInfos),(NeuralType[]){
        paddle1.p.y,
        paddle2.p.y,
        ball.p.y,
        ball.v.x,
        ball.v.y,
    },5);
    return neb;
}
void NeuralEnviroment_Func_Undo(NeuralEnviroment* ne,NeuralEnviromentBlock* eb){
    PongInfos* pi = (PongInfos*)eb->data;
    paddle1.p = pi->paddle1_p;
    paddle1.v = pi->paddle1_v;
    paddle2.p = pi->paddle2_p;
    paddle2.v = pi->paddle2_v;
    ball.p = pi->ball_p;
    ball.v = pi->ball_v;
}

void NeuralNode_Render(NeuralNode* n,int x,int y){
    RenderRect(x,y,100.0f,100.0f,GREEN);
    if(n){
        String str = String_Format("%f",n->value);
        RenderCStrSizeAlxFont(&infofont,str.Memory,str.size,x,y,GRAY);
        String_Free(&str);

        str = String_Format("%f",n->bias);
        RenderCStrSizeAlxFont(&infofont,str.Memory,str.size,x,y + infofont.CharSizeY,GRAY);
        String_Free(&str);

        str = String_Format("%f",n->cbias);
        RenderCStrSizeAlxFont(&infofont,str.Memory,str.size,x,y + 2 * infofont.CharSizeY,GRAY);
        String_Free(&str);

        for(int i = 0;i<n->weights.size;i++){
            NeuralWeight* nw = (NeuralWeight*)Vector_Get(&n->weights,i);
            
            str = String_Format("%f",nw->w);
            RenderCStrSizeAlxFont(&infofont,str.Memory,str.size,x + 150,y + i * infofont.CharSizeY,GRAY);
            String_Free(&str);
        }
    }
}
void NeuralNetwork_Render(NeuralNetwork* nn){
    for(int i = 0;i<nn->layers.size;i++){
        NeuralLayer* nl = (NeuralLayer*)Vector_Get(&nn->layers,i);
        
        for(int j = 0;j<nl->size;j++){
            NeuralNode* n = (NeuralNode*)Vector_Get(nl,j);
            if(n) NeuralNode_Render(n,i * 300.0f,j * 120.0f);
        }
    }
}

void* UpdateAi(void* arg){
    Thread* t = (Thread*)arg;

    int update = NN_TICKSUPDATE;
    while(t->running){
        if(update >= NN_TICKSUPDATE){
            printf("\033[2J\033[H");

            int state = NeuralNetwork_PlanDecision(&neuralnet);
            if(state<0){
                printf("Error! ");
                fflush(stdout);
            }else{
                NeuralType outs[3] = { 0.0f,0.0f,0.0f };
                outs[state] = 1.0f;

                NeuralNetwork_CalculateCosts(&neuralnet,outs);
                NeuralNetwork_ApplyCosts(&neuralnet,NN_LEARNRATE);

                printf("Updated! ");
                fflush(stdout);
            }
            aireward = 0.0f;
            update = 0;
        }else
            update++;

        Thread_Sleep_M(NN_DELAY);
        printf(".");
        fflush(stdout);
    }
    return NULL;
}

void Setup(AlxWindow* w){
    infofont = AlxFont_MAKE_YANIS(12,12);

    paddle1 = PongObject_New((Vec2){ PADDLE_1_POSX,0.0f },(Vec2){ PADDLE_WIDTH,PADDLE_HEIGHT },(Vec2){ 0.0f,0.0f },PADDLE_1_COLOR);
    paddle2 = PongObject_New((Vec2){ PADDLE_2_POSX,0.0f },(Vec2){ PADDLE_WIDTH,PADDLE_HEIGHT },(Vec2){ 0.0f,0.0f },PADDLE_2_COLOR);
    ball = PongObject_New((Vec2){ 0.0f,0.0f },(Vec2){ BALL_WIDTH,BALL_HEIGHT },(Vec2){ 0.0f,0.0f },BALL_COLOR);
    Reset();

    neuralnet = NeuralNetwork_MakeRL(
        NeuralEnviroment_New(
            NULL,
            (void*)NeuralEnviroment_Func_Step,
            (void*)NeuralEnviroment_Func_Block,
            (void*)NeuralEnviroment_Func_Undo
        ),
        (NeuralLayerCount[]){ 5,2,3,NeuralLayerCount_End }
    );

    updaterai = Thread_New(NULL,UpdateAi,&updaterai);
    Thread_Start(&updaterai);
}
void Update(AlxWindow* w){
    ReloadAlxFont(GetWidth() * 0.025f,GetHeight() * 0.05f);

    w->ElapsedTime = ElapTime;
    if(Stroke(ALX_KEY_Q).DOWN)  ElapTime *= 1.01f;
    if(Stroke(ALX_KEY_A).DOWN)  ElapTime *= 0.99f;

    //if(Stroke(ALX_KEY_W).DOWN)          paddle1.v.y = -PADDLE_SPEED;
    //else if(Stroke(ALX_KEY_S).DOWN)     paddle1.v.y = PADDLE_SPEED;
    //else                                paddle1.v.y = 0.0f;

    //if(Stroke(ALX_KEY_UP).DOWN)         paddle2.v.y = -PADDLE_SPEED;
    //else if(Stroke(ALX_KEY_DOWN).DOWN)  paddle2.v.y = PADDLE_SPEED;
    //else                                paddle2.v.y = 0.0f;

    float dy = (ball.p.y + ball.d.y * 0.5f) - (paddle1.p.y + paddle1.d.y * 0.5f);
    paddle1.v.y = PADDLE_SPEED * F32_Sign(dy);

    if(Stroke(ALX_KEY_SPACE).PRESSED){
        int state = NeuralNetwork_PlanDecision(&neuralnet);
        if(state<0){
            printf("Error!\n");
        }else{
            NeuralType outs[3] = { 0.0f,0.0f,0.0f };
            outs[state] = 1.0f;

            NeuralNetwork_CalculateCosts(&neuralnet,outs);
            NeuralNetwork_ApplyCosts(&neuralnet,NN_LEARNRATE);
        }
        aireward = 0.0f;
    }

    NeuralEnviromentBlock nb = NeuralEnviroment_Func_Block(&neuralnet.nenv);
    NeuralNetwork_SetInput(&neuralnet,nb.inputs);
    NeuralEnviromentBlock_Free(&nb);

    DecisionState d = NeuralNetwork_GetDecision(&neuralnet);
    if(d.state==0)      paddle2.v.y = -PADDLE_SPEED;
    else if(d.state==1) paddle2.v.y = PADDLE_SPEED;
    else if(d.state==2) paddle2.v.y = 0.0f;
    else                printf("Error!\n");
    DecisionState_Free(&d);

    PongObject_Update(&paddle1,w->ElapsedTime);
    PongObject_Update(&paddle2,w->ElapsedTime);
    PongObject_Update(&ball,w->ElapsedTime);
    
    PongObject_Collision(&paddle1,&ball);
    PongObject_Collision(&paddle2,&ball);

	Clear(BLACK);

    PongObject_Render(&paddle1);
    PongObject_Render(&paddle2);
    PongObject_Render(&ball);

    NeuralNetwork_Render(&neuralnet);

    String str = String_Format("T:%f, R:%f",ElapTime,aireward);
    RenderCStrSizeAlxFont(&infofont,str.Memory,str.size,GetWidth() - 500.0f,0.0f,WHITE);
    String_Free(&str);

    str = String_Format("%d : %d",score1,score2);
    RenderCStrSize(str.Memory,str.size,(GetWidth() - str.size * GetAlxFont()->CharSizeX) * 0.5f,0.0f,WHITE);
    String_Free(&str);
}
void Delete(AlxWindow* w){
    Thread_Stop(&updaterai);

	NeuralNetwork_Free(&neuralnet);
    AlxFont_Free(&infofont);
}

int main(){
    if(Create("Pong with AI",1920,1080,1,1,Setup,Update,Delete))
        Start();
    return 0;
}