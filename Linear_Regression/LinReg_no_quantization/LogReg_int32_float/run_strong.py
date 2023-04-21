import os 
import sys
import getpass

rootdir = "." # Include path to repo

applications = {"LRGD": ["NR_DPUS=X TYPE=Z make all", "./bin/host_code -i 500 -m M"]} 

def run(app_name):
    TYPE = ["INT32", "FLOAT"] 
    # NR_DPUS = [32, 64] 
    NR_DPUS = [256, 512, 1024, 2048] 
    # M_SZIE = [1024, 2048, 4096, 8192] 
    # M_SZIE = [163840, 327680, 491520, 655360, 819200] 
    M_SZIE = [6291456] # traing data size = 408MB/102MB for int32/int8

    if app_name in applications:
        print ("------------------------ Running: "+app_name+"----------------------")
        print ("--------------------------------------------------------------------")
        if(len(applications[app_name]) > 1):
            make = applications[app_name][0]
            run_cmd = applications[app_name][1]
        
            os.chdir(rootdir + "/")
            os.getcwd()
        
            os.system("make clean")

            try:
                os.mkdir(rootdir + "/"+ "bin")
            except OSError:
                print ("Creation of the direction /bin failed")

            try:
                os.mkdir(rootdir + "/profile")
            except OSError:
                print ("Creation of the direction /profile failed")
        
            for t in TYPE:
                for r in NR_DPUS:
                    os.system("make clean")
                    
                    m = make.replace("X", str(r))
                    m = m.replace("Z", str(t)) 
                    print ("Running = " + m) 
                    try:
                        os.system(m)
                    except: 
                        pass 
                    for m_size in M_SZIE: 
                        r_cmd = run_cmd.replace("M", str(m_size)) 
                        r_cmd = r_cmd +  " >> profile/outs_type"+str(t)+"_dpus"+str(r)+"_m"+str(m_size)
                        
                        print ("Running = " + app_name + " -> "+ r_cmd)
                        try:
                            os.system(r_cmd) 
                        except: 
                            pass 
        else:
            make = applications[app_name] 

            os.chdir(rootdir + "/"+app_name)
            os.getcwd()
        
            try:
                os.mkdir(rootdir + "/"+ app_name +"/bin")
                # os.mkdir(rootdir + "/"+ app_name +"/log")
                # os.mkdir(rootdir + "/"+ app_name +"/log/host")
                os.mkdir(rootdir + "/"+ app_name +"/profile")
            except OSError:
                print ("Creation of the direction failed")

            print (make)    
            os.system(make + ">& profile/out")

    else:
        print ( "Application "+app_name+" not available" )

def main():
    if(len(sys.argv) < 2):
        print ("Usage: python run.py application")
        print ("Applications available: ")
        for key, value in applications.iteritems():
            print (key )
        print ("All")

    else:
        cmd = sys.argv[1]
        print ("Application to run is: " + cmd )
        if cmd == "All":
            for key, value in applications.iteritems():
                run(key)
                os.chdir(rootdir)
        else:
            run(cmd)

if __name__ == "__main__":
    main()
