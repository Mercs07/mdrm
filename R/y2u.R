# function for reducing row-rank of Y (univariate first)
# approximately bin data such that the number of unique values
# is proportional to N^p.
Y2U = function(y,p=0.67,minB){
	if(p <= 0 | p >= 1) stop("parameter p must be between 0 and 1")
  if(NCOL(y)>1) return(y)
  y = y[is.finite(y)]
  N = length(y)
	nbin = round(N^p)
	if(missing(minB)) minB = max(2,round(0.1*N/nbin)) # bins small amount of data
	if(length(unique(y))<=nbin) return(y)
	# equivalent to ggplot2::cut_interval (but we need to track the breaks)
	yRng = range(y)
	yBR = seq(yRng[1],yRng[2],length.out = nbin+1)
	yI = cut(y,yBR,include.lowest = TRUE)
	nus = unname(sapply(split(y,yI),median))
	bincnt = table(yI)
	for(i in seq_along(levels(yI))){
	  li = levels(yI)[i]
	  if(bincnt[li] > minB){
	    y[yI==li] = nus[i]
	  }
	}
	y
}

# for mixed data, it is ineffective to simply bin the continuous outcome; 
# instead we should marginalize the data
# on values of the count variable and bin within these subsets
# the values 50 and .075 are just heuristics and could possibly be improved (consider the trade-offs between p and these numbers also)
mbin = function(disc,cont,p=0.5){
	uv = unique(disc)
	for(v in uv){
		ss = disc==v
		if(sum(ss) > 50 || mean(ss) > 0.075) cont[ss] = Y2U(cont[ss],p)
	}
	cont	
}

Y2U2 = function(Y,p=0.5,plot=FALSE){
	if(NCOL(Y)==1)     return(Y2U(Y,p))
	if(NCOL(Y)>2)      stop("Need a 2-column matrix.")
	if(p <= 0 | p > 1) stop("Pass a power between zero and one.")
	if(any(is.na(Y)))  stop("Please remove missing values first.")
	rr = apply(Y,2,range)
	uu = apply(Y,2,FUN=function(z){length(unique(z))})
	N = nrow(Y)
	gw = round(N^p)
	discTst = round(N^0.5) # heuristic to see if Y is 'discrete' enough

	if(max(uu) < discTst) U = Y # both vars. already discret(ized)
	else if(uu[1] < discTst) U = cbind(Y[,1],mbin(Y[,1],Y[,2],p))
	else if(uu[2] < discTst) U = cbind(mbin(Y[,2],Y[,1],p),Y[,2])
	else { # nether varaible is discrete - use hexbin
	  require(hexbin)
	  hexy = hexbin(Y,xbins = round(N^p),IDs = TRUE)
	  U = t(sapply(seq_along(hexy@cID),function(i){
	    cid = which(hexy@cID[i] == hexy@cell)
	    if(hexy@count[cid] == 1) Y[i,] else c(hexy@xcm[cid],hexy@ycm[cid])
	  }))
	}
	if(!is.null(colnames(Y))) colnames(U) = colnames(Y)
	if(plot){
	  GG = expand.grid(seq(rr[1,1],rr[2,1],length.out = gw),
	                   seq(rr[1,2],rr[2,2],length.out = gw))
	  op = par(no.readonly = TRUE); on.exit(par(op))
	  par(mar = c(3,3,3,1))
	  plot(GG,col = "#4c4b45",pch = ".",las = 1)
	  points(Y,col = "#f4ce42",pch = 4,cex = 1.1)
	  points(unique(U),pch = 19,col = "#5CACEEAB",cex = 1.2)
	  mtext(side = 3,line = 0.5,
	        substitute(paste(hat(rho)," = ",ch,", dim=",dU),
	                   list(ch = round(cor(U)[1,2],3),dU = dim(unique(U))[1])))
	}
	U
}